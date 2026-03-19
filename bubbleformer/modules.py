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

from bubbleformer.data.batching import CollatedBatch
from bubbleformer.data.normalize import get_normalizer
from bubbleformer.models import get_model
from bubbleformer.utils.lr_schedulers import CosineWarmupLR
#from bubbleformer.utils.plot_utils import wandb_sdf_plotter, wandb_temp_plotter, wandb_vel_plotter
from bubbleformer.layers.moe.topk_moe import TopkRouterWithBias

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
        self.normalizer_cfg = OmegaConf.to_container(normalizer_cfg, resolve=True)
        self.optimizer_cfg = OmegaConf.to_container(optim_cfg, resolve=True)
        self.scheduler_cfg = OmegaConf.to_container(scheduler_cfg, resolve=True)
        if normalization_constants is not None:
            self.normalization_constants = normalization_constants
        self.log_wandb = log_wandb

        self.normalizer = get_normalizer(self.normalizer_cfg)
        self.criterion = torch.nn.L1Loss()

        self.model_cfg["params"]["input_fields"] = len(self.data_cfg["input_fields"])
        self.model_cfg["params"]["output_fields"] = len(self.data_cfg["output_fields"])
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

    def default_log(self, key, value, **kwargs):
        kwargs["on_step"] = True
        kwargs["on_epoch"] = True
        kwargs["prog_bar"] = True
        kwargs["logger"] = True
        self.log(key, value, **kwargs)
        if self.log_wandb and self.trainer.is_global_zero:
            wandb.log({key: value})

    def default_log_dict(self, dict, **kwargs):
        kwargs["on_step"] = True
        kwargs["on_epoch"] = True
        kwargs["prog_bar"] = True
        kwargs["logger"] = True
        self.log_dict(dict, **kwargs)
        if self.log_wandb and self.trainer.is_global_zero:
            wandb.log(dict)

    def get_current_lr(self):
        opt = self.optimizers()
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
        if opt_name == "adamw":
            optimizer = AdamW(self.model.parameters(), **opt_params, fused=True)
        elif opt_name == "adam":
            optimizer = Adam(self.model.parameters(), **opt_params)
        elif opt_name == "lion":
            optimizer = Lion(self.model.parameters(), **opt_params)
        elif opt_name == "muon":
            optimizer = Muon(self.model.parameters(), **opt_params)
        else:
            raise ValueError(f"Optimizer {opt_name} not supported")

        scheduler_name = self.scheduler_cfg["name"]
        scheduler_params = self.scheduler_cfg["params"]
        if scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                            optimizer,
                            T_max=self.t_max,
                            eta_min=scheduler_params["eta_min"],
                            last_epoch=self.trainer.global_step - 1
                        )
        if scheduler_name == "cosine_warmup":
            scheduler = CosineWarmupLR(
                            optimizer,
                            warmup_iters=scheduler_params["warmup_iters"],
                            max_iters=self.t_max,
                            eta_min=scheduler_params["eta_min"],
                            last_epoch=self.trainer.global_step - 1
                        )
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def on_train_epoch_start(self):
        self.train_start_time = time.time()

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

            # perfect balance is 0, while 1 is imbalanced.
            coeff_of_variation = (tpe.std() / tpe.mean()).item()
            log_dict[f"{prefix}/coeff_of_variation_layer{moe_idx}"] = coeff_of_variation

            # Check the ratio of max load to the mean load.
            # Ideally, this metric should be close to 1.
            load_imbalance_factor = tpe.max() / tpe.mean()
            log_dict[f"{prefix}/load_imbalance_factor_layer{moe_idx}"] = load_imbalance_factor.item()

            # Check if any experts receive less than 1% of the tokens.
            # ideally, this metric should be 1.
            min_fraction = 0.01
            threshold = tpe.sum() * min_fraction
            active = (tpe > threshold).float().mean()
            log_dict[f"{prefix}/active_experts_layer{moe_idx}"] = active.item()
        return log_dict

    def training_step(
        self,
        batch: CollatedBatch,
        batch_idx: int
    ) -> torch.Tensor:
        batch = batch.normalize(self.normalizer)
        if random.random() < 0.5:
            batch = batch.fliplr()
        if random.random() < 0.9:
            batch = batch.noise(random.choice(torch.linspace(0.001, 1, 500).tolist()))

        inp = batch.get_input()
        pred, moe_outputs = self.model(inp)

        data_loss = self.criterion(pred, batch.target)

        # use router loss to do load balancing.
        router_with_loss = moe_outputs[0].router_output.router_type() == "loss"
        if router_with_loss:
            router_loss = sum(moe_output.router_output.load_balance_loss for moe_output in moe_outputs)
            loss = data_loss + router_loss
        else:
            loss = data_loss

        # using router bias to update the router.
        # TODO: perhaps this is better done in MoE module itself, during the forward pass?
        router_with_bias = moe_outputs[0].router_output.router_type() == "bias"
        if router_with_bias:
            router_idx = 0
            for module in self.modules():
                if isinstance(module, TopkRouterWithBias):
                    module.update_router_bias(moe_outputs[router_idx].router_output.tokens_per_expert)
                    router_idx += 1

        log_dict = {
            "train/loss": loss,
            "train/data_loss": data_loss,
            "train/learning_rate": self.get_current_lr()
        }
        if router_with_loss:
            log_dict["train/routing_loss"] = router_loss

        log_dict = self.moe_metrics(moe_outputs, log_dict, "train")

        # Simple confirmation that the standard deviation and mean normalized reasonably
        inp = inp.detach()
        log_dict["train/sdf_mean"] = inp.input[..., 0, :, :].mean().item()
        log_dict["train/temp_mean"] = inp.input[..., 1, :, :].mean().item()
        log_dict["train/velx_mean"] = inp.input[..., 2, :, :].mean().item()
        log_dict["train/vely_mean"] = inp.input[..., 3, :, :].mean().item()
        log_dict["train/sdf_std"] = inp.input[..., 0, :, :].std().item()
        log_dict["train/temp_std"] = inp.input[..., 1, :, :].std().item()
        log_dict["train/velx_std"] = inp.input[..., 2, :, :].std().item()
        log_dict["train/vely_std"] = inp.input[..., 3, :, :].std().item()

        self.default_log_dict(log_dict)

        return loss

    def validation_step(
        self,
        batch: CollatedBatch,
        batch_idx: int
    ) -> torch.Tensor:
        batch = batch.normalize(self.normalizer)
        inp = batch.get_input()
        pred, moe_outputs = self.model(inp)
        bulk_temp, _ = batch.get_temps()
        loss = self.criterion(pred, batch.target, bulk_temp)
        if batch_idx == 0:
            self.validation_sample = (batch.input.detach(), batch.target.detach(), pred.detach())

        log_dict = {
            "val/loss": loss,
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
