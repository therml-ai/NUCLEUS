import random
import time
from typing import Tuple, Optional, List
from math import sqrt

import wandb
from omegaconf import OmegaConf, DictConfig
import torch
from torch.optim import AdamW, Adam, Muon
from lion_pytorch import Lion
import lightning as L

from nucleus.data.batching import CollatedBatch
from nucleus.data.normalize import get_normalizer
# Import from the _api submodule (not the nucleus.models package) so this module
# can be imported from nucleus2_moe_divfree.py while the package __init__ is still
# executing -- DivFreeForecastModule lives there and subclasses a class defined here.
from nucleus.models._api import get_model, load_model_from_checkpoint
from nucleus.utils.lr_schedulers import CosineWarmupLR, TrapezoidalLR
from nucleus.layers.moe.topk_moe import TopkRouterWithBias
from nucleus.utils.losses import phase_bce_with_logits_loss, field_gradient_loss
from nucleus.utils.metrics import precision_recall
from nucleus.utils.physical_metrics import (
    eikonal,
    liquid_divergence,
    nucleation_event_masks,
)
from nucleus.noise import (
    LogUniformNoise,
    FieldDropout,
    FrameDropout,
    SpuriousBulkNucleation,
    BubbleResize,
    InterfaceJitter
)


class ModuleBase(L.LightningModule):
    def __init__(
        self,
        checkpoint_path: Optional[str],
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        normalizer_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        log_wandb: bool = False,
        normalization_constants: Optional[Tuple[List, List]] = None,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        self.data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
        self.normalizer_cfg = OmegaConf.to_container(normalizer_cfg, resolve=True)
        self.optimizer_cfg = OmegaConf.to_container(optim_cfg, resolve=True)
        self.scheduler_cfg = OmegaConf.to_container(scheduler_cfg, resolve=True)
        # Same normalizer the data pipeline uses. Subclasses may need to normalize/denormalize
        self.normalizer = get_normalizer(self.normalizer_cfg)
        self.save_hyperparameters(ignore=["model_cfg", "data_cfg", "normalizer_cfg", "optim_cfg", "scheduler_cfg"])
        if normalization_constants is not None:
            self.normalization_constants = normalization_constants
        self.log_wandb = log_wandb

        self.criterion = torch.nn.L1Loss()

        self.load_balance_loss_weight = self.model_cfg["params"].pop("load_balance_loss_weight", 1e-5)
        self.z_loss_weight = self.model_cfg["params"].pop("z_loss_weight", 1e-5)
        self.gradient_loss_weight = self.model_cfg["params"].pop("gradient_loss_weight", 1e-2)
        
        self.num_windows = self.model_cfg["params"].pop("num_windows", 3)

        if self.checkpoint_path is not None:
            self.model = load_model_from_checkpoint(self.checkpoint_path)
        else:
            self.model = get_model(self.model_cfg["name"], **self.model_cfg["params"])

        self.augmentations = [
            LogUniformNoise(0.001, 5e-2, skip_prob=0.1),
            FieldDropout(),
            #FrameDropout(),
        ]

        self.t_max = None
        self.validation_sample = None
        self.train_start_time = None
        self.val_start_time = None
        self._train_iter_prev_perf: Optional[float] = None

        if self.optimizer_cfg["name"] == "muon":
            self.automatic_optimization = False

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def default_log(self, key: str, value, **kwargs):
        kwargs["logger"] = True
        self.log(key, value, **kwargs)

    def default_log_dict(self, d: dict, **kwargs):
        kwargs["logger"] = True
        self.log_dict(d, **kwargs)

    def get_current_lr(self) -> float:
        opt = self.optimizers()
        if isinstance(opt, list):
            return opt[0].param_groups[0]["lr"]
        return opt.param_groups[0]["lr"]
    
    def log_step_metrics(self, log_dict: dict, pred: torch.Tensor, target: torch.Tensor, dx: float, dy: float, prefix: str):
        with torch.no_grad():
            return log_dict | {
                f"{prefix}/mae_loss": torch.nn.functional.l1_loss(pred, target),
                f"{prefix}/mse_loss": torch.nn.functional.mse_loss(pred, target),
                f"{prefix}/absmax_error": (pred - target).abs().max(),
                f"{prefix}/pred_mean": pred.mean(),
                f"{prefix}/pred_std": pred.std(),
                f"{prefix}/target_mean": target.mean(),
                f"{prefix}/target_std": target.std(),
                #f"{prefix}/eikonal_loss": (1 - eikonal(pred[..., 0], dx, dy)).abs().mean(),
                #f"{prefix}/liquid_divergence": liquid_divergence(pred[..., 2], pred[..., 3], pred[..., 0], dx, dy).mean(),
            }

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.t_max = self.trainer.estimated_stepping_batches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def _weight_decay_param_groups(self, weight_decay: float):
        # Only weight-decay >=2D weights (matmul / conv). 1D params -- biases,
        # LayerNorm/RMSNorm gains, scalar scales -- are excluded so weight decay
        # can't shrink them toward zero.
        decay, no_decay = [], []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            (decay if param.ndim >= 2 else no_decay).append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg["name"]
        opt_params = self.optimizer_cfg["params"]
        opt_params["lr"] = torch.tensor(opt_params["lr"])
        if opt_name == "adamw":
            weight_decay = opt_params.get("weight_decay", 0.0)
            adamw_kwargs = {key: value for key, value in opt_params.items() if key != "weight_decay"}
            param_groups = self._weight_decay_param_groups(weight_decay)
            optimizer = [AdamW(param_groups, **adamw_kwargs, fused=True)]
        elif opt_name == "adam":
            optimizer = [Adam(self.model.parameters(), **opt_params)]
        elif opt_name == "lion":
            optimizer = [Lion(self.model.parameters(), **opt_params)]
        elif opt_name == "muon":
            params2d = [p for p in self.model.parameters() if p.dim() == 2]
            params_other = [p for p in self.model.parameters() if p.dim() != 2]
            adamw = AdamW(params_other, **opt_params, fused=True)
            muon = Muon(params2d, **opt_params, adjust_lr_fn="match_rms_adamw")
            optimizer = [adamw, muon]
        else:
            raise ValueError(f"Optimizer {opt_name} not supported")

        scheduler_name = self.scheduler_cfg["name"]
        scheduler_params = self.scheduler_cfg["params"]
        if scheduler_name == "cosine_warmup":
            scheduler = [
                {
                    "scheduler": CosineWarmupLR(
                        optimizer[idx],
                        warmup_iters=scheduler_params["warmup"],
                        max_iters=self.t_max,
                        eta_min=scheduler_params["eta_min"],
                        last_epoch=self.trainer.global_step - 1,
                    ),
                    "interval": "step",
                    "frequency": 1,
                }
                for idx in range(len(optimizer))
            ]
        elif scheduler_name == "trapezoidal":
            warmup = scheduler_params["warmup"]
            cooldown = scheduler_params["cooldown"]
            if isinstance(warmup, float):
                warmup = warmup * self.t_max
            if isinstance(cooldown, float):
                cooldown = cooldown * self.t_max
            flat_iters = self.t_max - warmup - cooldown
            scheduler = [
                {
                    "scheduler": TrapezoidalLR(
                        optimizer[idx],
                        scale_factor=scheduler_params["scale_factor"],
                        warmup_iters=warmup,
                        flat_iters=flat_iters,
                        cooldown_iters=cooldown,
                        last_epoch=self.trainer.global_step - 1,
                    ),
                    "interval": "step",
                    "frequency": 1,
                }
                for idx in range(len(optimizer))
            ]
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported")

        return optimizer, scheduler

    def transfer_batch_to_device(self, batch: CollatedBatch, device: torch.device, dataloader_idx: int):
        batch = batch.pin_memory()
        return batch.to(device, non_blocking=True)

    def on_before_optimizer_step(self, optimizer):
        if self.global_step % 100 == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float("inf"),
            )
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)

    def get_noise_scale(self) -> float:
        if self.global_step < self.scheduler_cfg["params"]["warmup"]:
            return 0.0
        max_noise_scale = self.scheduler_cfg["params"].get("max_noise_scale", 1.0)
        if self.global_step < 10000:
            max_scale_at_step = max_noise_scale * (self.global_step / (self.t_max // 2))
            return abs(random.gauss(0, max_scale_at_step))
        return abs(random.gauss(0, max_noise_scale))

    def on_train_epoch_start(self):
        self.train_start_time = time.time()
        self._train_iter_prev_perf = None

    def on_train_batch_end(self, outputs, batch, batch_idx):
        now = time.perf_counter()
        if self._train_iter_prev_perf is not None:
            dt = now - self._train_iter_prev_perf
            if dt > 0 and self.trainer.is_global_zero:
                self.log(
                    "train/iteration_per_second",
                    1.0 / dt,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=False,
                )
        self._train_iter_prev_perf = now

    def on_train_epoch_end(self):
        if self.train_start_time is not None:
            train_time = time.time() - self.train_start_time
            if self.log_wandb and self.trainer.is_global_zero:
                wandb.log({"train/epoch_time": train_time, "epoch": self.current_epoch})

    def on_validation_epoch_start(self):
        self.val_start_time = time.time()
        if self.log_wandb and self.trainer.is_global_zero:
            try:
                train_loss = self.trainer.callback_metrics["train/loss"].item()
                wandb.log({"train/loss_epoch": train_loss, "epoch": self.current_epoch})
            except Exception:
                pass

    def on_validation_epoch_end(self):
        if self.val_start_time is not None:
            val_time = time.time() - self.val_start_time
            if self.log_wandb and self.trainer.is_global_zero:
                wandb.log({"val/epoch_time": val_time, "epoch": self.current_epoch})


class ConditionedForecastModule(ModuleBase):
    def training_step(self, batch: CollatedBatch, batch_idx: int) -> torch.Tensor:
        with torch.no_grad():
            for aug in self.augmentations:
                batch.input = aug(batch.input)

        inp = batch.get_input()
        torch.compiler.cudagraph_mark_step_begin()
        pred = self.model(inp)
        loss = self.criterion(pred, batch.target)

        log_dict = {"train/loss": loss, "train/learning_rate": self.get_current_lr()}
        log_dict = self.log_step_metrics(log_dict, pred, batch.target, batch, "train")
        self.default_log_dict(log_dict)
        return loss

    def validation_step(self, batch: CollatedBatch, batch_idx: int) -> torch.Tensor:
        inp = batch.get_input()
        torch.compiler.cudagraph_mark_step_begin()
        pred = self.model(inp)
        loss = self.criterion(pred, batch.target)

        log_dict = {"val/loss": loss}
        log_dict = self.log_step_metrics(log_dict, pred, batch.target, batch, "val")
        self.default_log_dict(log_dict)
        return loss


class MoEConditionedForecastModule(ModuleBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _moe_metrics(self, moe_outputs: list, log_dict: dict, prefix: str) -> dict:
        for moe_idx, moe_output in enumerate(moe_outputs):
            if hasattr(moe_output, "router_output"):
                tpe = moe_output.router_output.tokens_per_expert.float()
                mean_router_logit = moe_output.router_output.router_logits.mean()
                max_router_logit = moe_output.router_output.router_logits.abs().max()
            else:
                tpe = moe_output.tokens_per_expert.float()
                mean_router_logit = moe_output.router_logits.mean()
                max_router_logit = moe_output.router_logits.abs().max()

            log_dict[f"{prefix}_moe/mean_router_logit_layer{moe_idx}"] = mean_router_logit
            log_dict[f"{prefix}_moe/max_router_logit_layer{moe_idx}"] = max_router_logit
            coeff_of_variation = tpe.std() / tpe.mean()
            log_dict[f"{prefix}_moe/coeff_of_variation_layer{moe_idx}"] = coeff_of_variation
            load_imbalance_factor = tpe.max() / tpe.mean()
            log_dict[f"{prefix}_moe/load_imbalance_factor_layer{moe_idx}"] = load_imbalance_factor
            threshold = tpe.sum() * 0.01
            log_dict[f"{prefix}_moe/active_experts_layer{moe_idx}"] = (tpe > threshold).float().mean()
        return log_dict

    def _router_loss(self, moe_outputs: list) -> Tuple[torch.Tensor, bool]:
        """Returns (auxiliary_loss, router_has_loss) for the batch."""
        if not hasattr(moe_outputs[0], "router_output"):
            load_balance_loss = sum(o.load_balance_loss for o in moe_outputs)
            return load_balance_loss * self.load_balance_loss_weight, False

        router_type = moe_outputs[0].router_output.router_type()
        assert router_type in ("loss", "bias")

        load_balance_loss = sum(o.router_output.load_balance_loss for o in moe_outputs)
        z_loss = sum(o.router_output.z_loss for o in moe_outputs)
        aux_loss = (load_balance_loss * self.load_balance_loss_weight
                    + z_loss * self.z_loss_weight)
        return aux_loss, True

    def _update_router_bias(self, moe_outputs: list):
        if not hasattr(moe_outputs[0], "router_output"):
            return
        if moe_outputs[0].router_output.router_type() != "bias":
            return
        router_idx = 0
        for module in self.modules():
            if isinstance(module, TopkRouterWithBias):
                module.update_router_bias(moe_outputs[router_idx].router_output.tokens_per_expert)
                router_idx += 1

    def training_step(self, batch: CollatedBatch, batch_idx: int) -> torch.Tensor:
        with torch.no_grad():
            for aug in self.augmentations:
                batch.input = aug(batch.input)

        inp = batch.get_input()
        torch.compiler.cudagraph_mark_step_begin()
        pred, moe_outputs = self.model(inp)

        data_loss = self.criterion(pred, batch.target)
        aux_loss, router_has_loss = self._router_loss(moe_outputs)
        loss = data_loss + aux_loss

        self._update_router_bias(moe_outputs)

        log_dict = {
            "train/loss": loss,
            "train/data_loss": data_loss,
            "train/step": self.global_step,
            "train/learning_rate": self.get_current_lr(),
        }
        log_dict = self.log_step_metrics(log_dict, pred, batch.target, batch.dx, batch.dy, "train")
        log_dict = self._moe_metrics(moe_outputs, log_dict, "train")
        self.default_log_dict(log_dict)
        return loss

    def validation_step(self, batch: CollatedBatch, batch_idx: int) -> torch.Tensor:
        inp = batch.get_input()
        pred, moe_outputs = self.model(inp)
        loss = self.criterion(pred, batch.target)
        if batch_idx == 0:
            self.validation_sample = (batch.input.detach(), batch.target.detach(), pred.detach())

        log_dict = { "val/loss": loss }
        log_dict = self.log_step_metrics(log_dict, pred, batch.target, batch.dx[0].item(), batch.dy[0].item(), "val")
        log_dict = self._moe_metrics(moe_outputs, log_dict, "val")
        self.default_log_dict(log_dict)
        return loss
    
class PhaseForecastModule(MoEConditionedForecastModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_augmentations = [
            SpuriousBulkNucleation(),
            BubbleResize(),
            InterfaceJitter()
        ]
        
    def _sdf_to_phase(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sdf = tensor[..., 0]
        phase = (sdf > 0).to(torch.int32)
        fields = tensor[..., 1:]
        return fields, phase
    
    def _phase_precision_recall(self, input_phase, target_phase, pred_phase_logits, prefix):
        pred_phase = (pred_phase_logits > 0).to(torch.int32)
        # The frame preceding each target frame: the last input frame, followed
        # by the earlier target frames. Used as the liquid reference for both the
        # ground truth and the prediction so nucleation recall is measured over
        # the cells that were actually liquid beforehand.
        prev_phase = torch.cat((input_phase[:, -1:], target_phase[:, :-1]), dim=1)
        gt_nucleation, pred_nucleation = nucleation_event_masks(prev_phase, target_phase, pred_phase)

        nucleation_precision, nucleation_recall = precision_recall(pred_nucleation, gt_nucleation)
        vapor_precision, vapor_recall = precision_recall(pred_phase, target_phase)
        return {
            f"{prefix}/nucleation_precision": nucleation_precision,
            f"{prefix}/nucleation_recall": nucleation_recall,
            f"{prefix}/vapor_precision": vapor_precision,
            f"{prefix}/vapor_recall": vapor_recall,
        }
        
    def forward(self, batch: CollatedBatch):
        fields, phase = self._sdf_to_phase(batch.input)
        return self.model(fields, phase, batch.sim_params_tensor)
    
    def training_step(self, batch: CollatedBatch, batch_idx: int):
        fields, phase = self._sdf_to_phase(batch.input)
        augmented_phase = phase.clone()
        if self.global_step > self.scheduler_cfg["params"]["warmup"]:
            with torch.no_grad():
                for aug in self.augmentations:
                    fields = aug(fields)
                for aug in self.phase_augmentations:
                    augmented_phase = aug(augmented_phase)
                
        pred_fields, pred_phase_logits, moe_outputs = self.model.step(fields, augmented_phase, batch.sim_params_tensor)
        target_fields, target_phase = self._sdf_to_phase(batch.target)
        
        field_loss = self.criterion(pred_fields, target_fields)
        phase_loss = phase_bce_with_logits_loss(phase, target_phase, pred_phase_logits, 1.5, sqrt(20))
        gradient_loss = self.gradient_loss_weight * field_gradient_loss(pred_fields, target_fields)
        data_loss = field_loss + phase_loss + gradient_loss

        aux_loss, router_has_loss = self._router_loss(moe_outputs)
        loss = data_loss + aux_loss
        self._update_router_bias(moe_outputs)

        log_dict = {
            "train/loss": loss,
            "train/field_loss": field_loss,
            "train/phase_loss": phase_loss,
            "train/gradient_loss": gradient_loss,
            "train/data_loss": data_loss,
            "train/step": self.global_step,
            "train/learning_rate": self.get_current_lr(),
        }
        log_dict |= self._phase_precision_recall(phase, target_phase, pred_phase_logits, "train")
        log_dict = self._moe_metrics(moe_outputs, log_dict, "train")
        self.default_log_dict(log_dict)
        return loss

    def validation_step(self, batch: CollatedBatch, batch_idx: int) -> torch.Tensor:
        fields, phase = self._sdf_to_phase(batch.input)
        pred_fields, pred_phase_logits, moe_outputs = self.model.step(fields, phase, batch.sim_params_tensor)
        target_fields, target_phase = self._sdf_to_phase(batch.target)

        field_loss = self.criterion(pred_fields, target_fields)
        phase_loss = phase_bce_with_logits_loss(phase, target_phase, pred_phase_logits, 2.0, 20.0)
        gradient_loss = self.gradient_loss_weight * field_gradient_loss(pred_fields, target_fields)
        loss = field_loss + phase_loss

        log_dict = {
            "val/loss": loss,
            "val/field_loss": field_loss,
            "val/phase_loss": phase_loss,
            "val/gradient_loss": gradient_loss,
        }
        log_dict |= self._phase_precision_recall(phase, target_phase, pred_phase_logits, "val")
        log_dict = self._moe_metrics(moe_outputs, log_dict, "val")
        self.default_log_dict(log_dict)
        return loss

def get_train_module(module_name: str):
    if module_name == "conditioned_forecast":
        return ConditionedForecastModule
    elif module_name == "moe_conditioned_forecast":
        return MoEConditionedForecastModule
    elif module_name == "phase_forecast_module":
        return PhaseForecastModule
    elif module_name == "divfree_forecast":
        # Imported lazily: DivFreeForecastModule lives in nucleus2_moe_divfree.py
        # and subclasses MoEConditionedForecastModule (defined here), so importing
        # it at module top would be a circular import.
        from nucleus.models.nucleus2_moe_divfree import DivFreeForecastModule
        return DivFreeForecastModule
    else:
        raise ValueError(f"Module {module_name} not supported")
