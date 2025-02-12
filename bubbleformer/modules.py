import random
from typing import Tuple, Optional

import wandb
from omegaconf import OmegaConf, DictConfig
import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import lightning as L

from bubbleformer.models import get_model
from bubbleformer.utils.losses import LpLoss
from bubbleformer.utils.lr_schedulers import CosineWarmupLR
from bubbleformer.utils.plot_utils import wandb_sdf_plotter, wandb_temp_plotter, wandb_vel_plotter


class ForecastModule(L.LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        log_wandb: bool = False,
        normalization_constants: Tuple[torch.Tensor, torch.Tensor] = None
    ):
        super().__init__()
        self.model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        self.data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
        self.optimizer_cfg = OmegaConf.to_container(optim_cfg, resolve=True)
        self.scheduler_cfg = OmegaConf.to_container(scheduler_cfg, resolve=True)
        if normalization_constants is not None:
            self.normalization_constants = normalization_constants
        self.log_wandb = log_wandb

        self.model_cfg["params"]["fields"] = len(self.data_cfg["fields"])
        self.save_hyperparameters()

        self.criterion = LpLoss(d=2, p=2, reduce_dims=[0,1,2], reductions=["mean", "mean", "sum"])
        self.model = get_model(self.model_cfg["name"], **self.model_cfg["params"])
        self.T_max = None
        self.validation_sample = None
    
    def setup(
        self,
        stage: Optional[str] = None
    ):
        if stage == "fit":
            self.T_max = self.trainer.estimated_stepping_batches

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

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        self.log(
            "learning_rate",
            current_lr,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True
        )
        if self.log_wandb:
            wandb.log({"train_loss": loss, "learning_rate": current_lr})

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        inp, tgt = batch
        pred = self.model(inp)
        loss = self.criterion(pred, tgt)
        if random.random() < 0.5:
            self.validation_sample = (inp.detach(), tgt.detach(), pred.detach())

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        if self.log_wandb:
            wandb.log({"val_loss": loss})

        return loss

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg["name"]
        opt_params = self.optimizer_cfg["params"]
        if opt_name == "adamw":
            optimizer = AdamW(self.model.parameters(), **opt_params)
        elif opt_name == "adam":
            optimizer = Adam(self.model.parameters(), **opt_params)
        else:
            raise ValueError(f"Optimizer {opt_name} not supported")

        scheduler_name = self.scheduler_cfg["name"]
        scheduler_params = self.scheduler_cfg["params"]
        if scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                            optimizer,
                            T_max=self.T_max,
                            eta_min=scheduler_params["eta_min"],
                            last_epoch=self.trainer.global_step - 1
                        )
        if scheduler_name == "cosine_warmup":
            scheduler = CosineWarmupLR(
                            optimizer,
                            warmup_iters=scheduler_params["warmup_iters"],
                            max_iters=self.T_max,
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

    def on_validation_epoch_end(self):
        fields = self.data_cfg["fields"]
        if self.validation_sample is None:
            return
        inputs, targets, predictions = self.validation_sample
        
        #input_sample = inputs[0]  # T, C, H, W
        target_sample = targets[0] # T, C, H, W
        pred_sample = predictions[0] # T, C, H, W

        if self.log_wandb:
            try:
                sdf_idx = fields.index("dfun")
                #input_sdfs = wandb_sdf_plotter(input_sample[:,sdf_idx,:,:])
                target_sdfs = wandb_sdf_plotter(target_sample[:,sdf_idx,:,:])
                pred_sdfs = wandb_sdf_plotter(pred_sample[:,sdf_idx,:,:])
                wandb.log({
                    #"Input SDF": wandb.Image(input_sdfs),
                    "Target SDF": wandb.Image(target_sdfs, caption=f"Epc {self.current_epoch}"),
                    "Prediction SDF": wandb.Image(pred_sdfs, caption=f"Epc {self.current_epoch}"),
                })

            except ValueError:
                pass
            try:
                temp_idx = fields.index("temperature")
                #input_temps = wandb_temp_plotter(input_sample[:,temp_idx,:,:])
                target_temps = wandb_temp_plotter(target_sample[:,temp_idx,:,:])
                pred_temps = wandb_temp_plotter(pred_sample[:,temp_idx,:,:])
                wandb.log({
                    #"Input Temperature": wandb.Image(input_temps),
                    "Target Temp": wandb.Image(target_temps, caption=f"Epc {self.current_epoch}"),
                    "Prediction Temp": wandb.Image(pred_temps, caption=f"Epc {self.current_epoch}")
                })
            except ValueError:
                pass
            try:
                velx_idx = fields.index("velx")
                vely_idx = fields.index("vely")
                #input_vel_field = torch.stack([
                #                        input_sample[:,velx_idx,:,:],
                #                        input_sample[:,vely_idx,:,:]
                #                    ],
                #                    dim=1
                #                )
                target_vel_field = torch.stack([
                                        target_sample[:,velx_idx,:,:],
                                        target_sample[:,vely_idx,:,:]
                                    ],
                                    dim=1
                                )
                pred_vel_field = torch.stack([
                                        pred_sample[:,velx_idx,:,:],
                                        pred_sample[:,vely_idx,:,:]
                                    ],
                                    dim=1
                                )
                #input_vels = wandb_vel_plotter(input_vel_field)
                target_vels = wandb_vel_plotter(target_vel_field)
                pred_vels = wandb_vel_plotter(pred_vel_field)
                wandb.log({
                    #"Input Velocity": wandb.Image(input_vels),
                    "Target Vel": wandb.Image(target_vels, caption=f"Epc {self.current_epoch}"),
                    "Prediction Vel": wandb.Image(pred_vels, caption=f"Epc {self.current_epoch}")
                })
            except ValueError:
                pass

        plt.close("all")
        self.validation_outputs = []
