import random
import time
from typing import Tuple, Optional, List

import wandb
from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
import torch
from torch.optim import AdamW, Adam, Muon
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import lightning as L

from nucleus.data.batching import CollatedBatch
from nucleus.data.normalize import get_normalizer
from nucleus.models import get_model
from nucleus.utils.lr_schedulers import CosineWarmupLR, TrapezoidalLR
#from nucleus.utils.plot_utils import wandb_sdf_plotter, wandb_temp_plotter, wandb_vel_plotter
from nucleus.layers.moe.topk_moe import TopkRouterWithBias

class ForecastModule(L.LightningModule):
    """
    Module for training forecasting models with equal
    input and output time windows.
    Args:
        model_cfg (DictConfig): YAML Model config loaded using OmegaConf
        data_cfg (DictConfig): YAML Data config loaded using OmegaConf
        optim_cfg (DictConfig): YAML Optimizer config loaded using OmegaConf
        scheduler_cfg (DictConfig): YAML Scheduler config loaded using OmegaConf
        log_wandb (bool): Whether to log to wandb
        normalization_constants (Tuple[List, List]):
                    Difference and Division constants for normalization
    """
    def __init__(
        self,
        checkpoint_path: Optional[str],
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        normalizer_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        log_wandb: bool = False,
        normalization_constants: Tuple[List, List] = None
    ):
        super().__init__()
        # whole model config to be saved to the checkpoint
        self.save_hyperparameters()

        self.checkpoint_path = checkpoint_path
        self.model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        self.data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
        self.optimizer_cfg = OmegaConf.to_container(optim_cfg, resolve=True)
        self.scheduler_cfg = OmegaConf.to_container(scheduler_cfg, resolve=True)
        if normalization_constants is not None:
            self.normalization_constants = normalization_constants
        self.log_wandb = log_wandb

        self.criterion = torch.nn.L1Loss()

        self.load_balance_loss_weight = self.model_cfg["params"].get("load_balance_loss_weight")
        self.z_loss_weight = self.model_cfg["params"].get("z_loss_weight")

        self.model_cfg["params"]["input_fields"] = len(self.data_cfg["input_fields"])
        self.model_cfg["params"]["output_fields"] = len(self.data_cfg["output_fields"])
        del self.model_cfg["params"]["load_balance_loss_weight"]
        del self.model_cfg["params"]["z_loss_weight"]
        self.model = get_model(self.model_cfg["name"], **self.model_cfg["params"])
        if self.checkpoint_path is not None:
            model_data = torch.load(self.checkpoint_path, weights_only=False)
            weight_state_dict = OrderedDict()
            for key, val in model_data["state_dict"].items():
                #name = key[6:]
                weight_state_dict[key] = val
            del model_data
            self.load_state_dict(weight_state_dict)

        self.save_hyperparameters()
        self.t_max = None
        self.validation_sample = None
        self.train_start_time = None
        self.val_start_time = None
        self._train_iter_prev_perf: Optional[float] = None

        # If we're using Muon, we need two optimizers Muon for 2d
        # parameters, and AdamW for everything else. Using multiple
        # optimizers requires manual optimization.
        if self.optimizer_cfg["name"] == "muon":
           self.automatic_optimization = False

    def default_log(self, key, value, **kwargs):
        kwargs["logger"] = True
        self.log(key, value, **kwargs)

    def default_log_dict(self, dict, **kwargs):
        kwargs["logger"] = True
        self.log_dict(dict, **kwargs)

    def get_current_lr(self):
        opt = self.optimizers()
        if isinstance(opt, list):
            return opt[0].param_groups[0]['lr']
        else:
            return opt.param_groups[0]['lr']

    def setup(
        self,
        stage: Optional[str] = None
    ):
        if stage == "fit":
            self.t_max = self.trainer.estimated_stepping_batches

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        inp, tgt = batch
        pred = self.model(inp)
        loss = self.criterion(pred, tgt)

        self.default_log_dict({
            "train/loss": loss,
            "train/learning_rate": self.get_current_lr()
        })

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        inp, tgt = batch
        pred = self.model(inp)
        loss = self.criterion(pred, tgt)
        if batch_idx == 0:
            self.validation_sample = (inp.detach(), tgt.detach(), pred.detach())

        self.default_log_dict({"val/loss": loss})

        return loss

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg["name"]
        opt_params = self.optimizer_cfg["params"]
        opt_params["lr"] = torch.tensor(opt_params["lr"]) # wrap in tensor to avoid recompiles
        if opt_name == "adamw":
            optimizer = [AdamW(self.model.parameters(), **opt_params, fused=True)]
        elif opt_name == "adam":
            optimizer = [Adam(self.model.parameters(), **opt_params)]
        elif opt_name == "lion":
            optimizer = [Lion(self.model.parameters(), **opt_params)]
        elif opt_name == "muon":
            # Use Muon for 2D parameters and AdamW for everything else.
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
            scheduler = [CosineWarmupLR(
                            optimizer,
                            warmup_iters=scheduler_params["warmup_iters"],
                            max_iters=self.t_max,
                            eta_min=scheduler_params["eta_min"],
                            last_epoch=self.trainer.global_step - 1
                        )]
        elif scheduler_name == "trapezoidal":
            # warmup and cooldown are percentages if floats. Otherwise, total steps.
            warmup = scheduler_params["warmup"]
            cooldown = scheduler_params["cooldown"]
            if isinstance(warmup, float):
                warmup = warmup * self.t_max
            if isinstance(cooldown, float):
                cooldown = cooldown * self.t_max
            flat_iters = self.t_max - warmup - cooldown
            scheduler = [{
                    "scheduler": TrapezoidalLR(
                        optimizer[idx],
                        scale_factor=scheduler_params["scale_factor"],
                        warmup_iters=warmup,
                        flat_iters=flat_iters,
                        cooldown_iters=cooldown,
                        last_epoch=self.trainer.global_step - 1
                    ),
                    "interval": "step",
                    "frequency": 1
                } for idx in range(len(optimizer))
            ]
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported")

        return optimizer, scheduler

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
        if self.train_start_time is not None: # when resuming from middle of epoch, var is None
            train_time = time.time() - self.train_start_time
            if self.log_wandb and self.trainer.is_global_zero:
                wandb.log({"train/epoch_time": train_time, "epoch": self.current_epoch})

    def on_validation_epoch_start(self):
        self.val_start_time = time.time()
        if self.log_wandb and self.trainer.is_global_zero:
            try:
                train_loss = self.trainer.callback_metrics["train/loss"].item()
                wandb.log({"train/loss_epoch": train_loss, "epoch": self.current_epoch})
            except:
                pass

    def on_validation_epoch_end(self):
        if self.val_start_time is not None:
            val_time = time.time() - self.val_start_time
            if self.log_wandb and self.trainer.is_global_zero:
                wandb.log({"val/epoch_time": val_time, "epoch": self.current_epoch})

class ConditionedForecastModule(ForecastModule):
    """
    Module for training forecasting models with different
    input and output time windows.
    Args:
        model_cfg (DictConfig): YAML Model config loaded using OmegaConf
        data_cfg (DictConfig): YAML Data config loaded using OmegaConf
        optim_cfg (DictConfig): YAML Optimizer config loaded using OmegaConf
        scheduler_cfg (DictConfig): YAML Scheduler config loaded using OmegaConf
        log_wandb (bool): Whether to log to wandb
        normalization_constants (Tuple[List, List]):
                    Difference and Division constants for normalization
    """
    def __init__(
        self,
        checkpoint_path: str,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        normalizer_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        log_wandb: bool = False,
        normalization_constants: Tuple[List, List] = None
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            normalizer_cfg=normalizer_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            log_wandb=log_wandb,
            normalization_constants=normalization_constants
        )

    def training_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        batch = batch.normalize(self.normalizer)
        if random.random() < 0.5:
            batch = batch.fliplr()
        if random.random() < 0.8:
            sdf_scale = random.choice(torch.linspace(0.001, 1, 500).tolist())
            temp_scale = random.choice(torch.linspace(0.001, 5, 500).tolist())
            vel_scale = random.choice(torch.linspace(0.001, 0.3, 500).tolist())
            batch = batch.gaussian_noise(sdf_scale, temp_scale, vel_scale)

        inp = batch.get_input()
        pred = self.model(inp)
        bulk_temp, _ = batch.get_temps()
        loss = self.criterion(pred, batch.target, bulk_temp)

        self.default_log_dict({
            "train_loss": loss,
            "learning_rate": self.get_current_lr()
        })

        return loss

    def validation_step(
        self,
        batch: CollatedBatch,
        batch_idx: int
    ) -> torch.Tensor:
        batch = batch.normalize(self.normalizer)
        inp = batch.get_input()
        pred = self.model(inp)
        bulk_temp, _ = batch.get_temps()
        loss = self.criterion(pred, batch.target, bulk_temp)
        if batch_idx == 0:
            self.validation_sample = (batch.input.detach(), batch.target.detach(), pred.detach())

        self.default_log_dict({"val_loss": loss})

        return loss

class MoEConditionedForecastModule(ConditionedForecastModule):
    def __init__(
        self,
        checkpoint_path: str,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        normalizer_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        log_wandb: bool = False,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            normalizer_cfg=normalizer_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            log_wandb=log_wandb,
        )

    def moe_metrics(self, moe_outputs, log_dict: dict, prefix: str) -> dict:
        for moe_idx, moe_output in enumerate(moe_outputs):
            tpe = moe_output.router_output.tokens_per_expert.float()

            # check the mean router logit
            mean_router_logit = moe_output.router_output.router_logits.mean()
            log_dict[f"{prefix}_moe/mean_router_logit_layer{moe_idx}"] = mean_router_logit.item()

            # check the max router logit
            max_router_logit = moe_output.router_output.router_logits.abs().max()
            log_dict[f"{prefix}_moe/max_router_logit_layer{moe_idx}"] = max_router_logit.item()

            # perfect balance is 0, while 1 is imbalanced.
            coeff_of_variation = (tpe.std() / tpe.mean()).item()
            log_dict[f"{prefix}_moe/coeff_of_variation_layer{moe_idx}"] = coeff_of_variation

            # Check the ratio of max load to the mean load.
            # Ideally, this metric should be close to 1.
            load_imbalance_factor = tpe.max() / tpe.mean()
            log_dict[f"{prefix}_moe/load_imbalance_factor_layer{moe_idx}"] = load_imbalance_factor.item()

            # Check if any experts receive less than 1% of the tokens.
            # ideally, this metric should be 1.
            min_fraction = 0.01
            threshold = tpe.sum() * min_fraction
            active = (tpe > threshold).float().mean()
            log_dict[f"{prefix}_moe/active_experts_layer{moe_idx}"] = active.item()
        return log_dict

    def on_before_optimizer_step(self, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=float("inf"),  # not clipping. Only used to get grad norm.
        )
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)

    def transfer_batch_to_device(self, batch: CollatedBatch, device: torch.device, dataloader_idx: int):
        r"""
        Since our batch is in a dataclass, pytorch and lightning cannot figure out how to pin memory and
        asynchrously transfer the batch to the device. So, we do this manually.
        """
        batch.fluid_params_tensor = batch.get_fluid_params_tensor('cpu')
        pinned_batch = batch.pin_memory()
        return pinned_batch.to(device, non_blocking=True)

    def get_noise_scale(self):
        # During learning rate warmup, no noise is added.
        if self.global_step < self.scheduler_cfg["params"]["warmup"]:
            return 0.0
        max_noise_scale = 1.0
        # ramp up noise scale in first half of training.
        if self.global_step < self.t_max // 2:
            max_scale_at_step = max_noise_scale * (self.global_step / (self.t_max // 2))
            return random.uniform(0, max_scale_at_step)
        else:
            return random.uniform(0, max_noise_scale)

    def training_step(
        self,
        batch: CollatedBatch,
        batch_idx: int
    ) -> torch.Tensor:
    
        with torch.no_grad():
            batch.noise_(self.get_noise_scale())
            
        inp = batch.get_input()
        torch.compiler.cudagraph_mark_step_begin()
        pred, moe_outputs = self.model(inp)

        data_loss = self.criterion(pred, batch.target)

        # use router loss to do load balancing.
        router_with_loss = moe_outputs[0].router_output.router_type() in ("loss", "bias")
        if router_with_loss:
            router_load_balance_loss = sum(moe_output.router_output.load_balance_loss for moe_output in moe_outputs)
            router_z_loss = sum(moe_output.router_output.z_loss for moe_output in moe_outputs)
            loss = data_loss + (router_load_balance_loss * self.load_balance_loss_weight) + (router_z_loss * self.z_loss_weight)
        else:
            loss = data_loss

        # using router bias to update the router.
        router_with_bias = moe_outputs[0].router_output.router_type() == "bias"
        if router_with_bias:
            router_idx = 0
            for module in self.modules():
                if isinstance(module, TopkRouterWithBias):
                    module.update_router_bias(moe_outputs[router_idx].router_output.tokens_per_expert)
                    router_idx += 1

        if not self.automatic_optimization:
            # MANUAL OPTIMIZATION IF USING MULTIPLE OPTIMIZERS
            optimizers = self.optimizers()
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
            for opt in optimizers:
                opt.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            for opt in optimizers:
                opt.step()
            # global_step is incremented by the number of optimizers.
            # Subtracting to get the actual training step.
            self.global_step -= len(optimizers) - 1

            # MANUALLY APPLY SCHEDULER
            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            for scheduler in schedulers:
                scheduler.step()
                
        mse_loss = torch.nn.functional.mse_loss(pred.detach(), batch.target.detach())

        log_dict = {
            "train/loss": loss,
            "train/data_loss": data_loss,
            "train/mse_loss": mse_loss,
            "train/step": self.global_step,
            "train/learning_rate": self.get_current_lr(),
        }
        if router_with_loss:
            log_dict["train_moe/load_balance_loss"] = router_load_balance_loss
            log_dict["train_moe/z_loss"] = router_z_loss

        # compute expensive metrics less frequently--has non-trivial runtime overhead.
        if self.global_step % 100 == 0:
            with torch.no_grad():
                log_dict["train/input_mean"] = inp.input.mean().item()
                log_dict["train/input_std"] = inp.input.std().item()
                log_dict["train/target_mean"] = batch.target.mean().item()
                log_dict["train/target_std"] = batch.target.std().item()
                log_dict["train/pred_mean"] = pred.mean().item()
                log_dict["train/pred_std"] = pred.std().item()
                log_dict = self.moe_metrics(moe_outputs, log_dict, "train")

        self.default_log_dict(log_dict)

        return loss

    def validation_step(
            self,
        batch: CollatedBatch,
        batch_idx: int
    ) -> torch.Tensor:
        inp = batch.get_input()
        pred, moe_outputs = self.model(inp)
        loss = self.criterion(pred, batch.target)
        if batch_idx == 0:
            self.validation_sample = (batch.input.detach(), batch.target.detach(), pred.detach())

        mse_loss = torch.nn.functional.mse_loss(pred.detach(), batch.target.detach())

        log_dict = {
            "val/loss": loss,
            "val/mse_loss": mse_loss,

        }
        log_dict = self.moe_metrics(moe_outputs, log_dict, "val")
        self.default_log_dict(log_dict)

        return loss

def get_train_module(module_name: str):
    if module_name == "forecast":
        return ForecastModule
    elif module_name == "conditioned_forecast":
        return ConditionedForecastModule
    elif module_name == "moe_conditioned_forecast":
        return MoEConditionedForecastModule
    else:
        raise ValueError(f"Module {module_name} not supported")
