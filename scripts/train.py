import os
import pprint

import hydra
import wandb
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import CSVLogger

from bubbleformer.data import BubblemlForecast
from bubbleformer.modules import ForecastModule


@hydra.main(version_base=None, config_path="../bubbleformer/config", config_name="default")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    params = {}
    params["distributed"] = cfg.distributed
    params["checkpoint_path"] = cfg.checkpoint_path
    params["data_cfg"] = cfg.data_cfg
    params["model_cfg"] = cfg.model_cfg
    params["optim_cfg"] =  cfg.optim_cfg
    params["scheduler_cfg"] =  cfg.scheduler_cfg

    log_id = (
        cfg.model_cfg.name.lower() + "_"
        + cfg.data_cfg.dataset.lower() + "_"
        + os.getenv("SLURM_JOB_ID")
    )
    params["log_dir"] = os.path.join(cfg.log_dir, log_id)
    os.makedirs(params["log_dir"], exist_ok=True)
    
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(params)

    logger = CSVLogger(save_dir=params["log_dir"])
    wandb_run = None
    if cfg.use_wandb:
        try:
            wandb_key_path = "bubbleformer/config/wandb_api_key.txt"
            with open(wandb_key_path, "r", encoding="utf-8") as f:
                wandb_key = f.read().strip()
            wandb.login(key=wandb_key)
            wandb_run = wandb.init(
                project="bubbleformer",
                name=log_id,
                dir=params["log_dir"],
                tags=cfg.wandb_tags,
                config=params
            )
        except FileNotFoundError as e:
            print(e)
            print("Valid wandb API key not found at path src/config/wandb_api_key.txt")

    train_dataset = BubblemlForecast(
                filenames=cfg.data_cfg.train_paths,
                fields=cfg.data_cfg.fields,
                norm=cfg.data_cfg.normalize,
                time_window=cfg.data_cfg.time_window,
            )
    normalization_constants = train_dataset.normalize()
    val_dataset = BubblemlForecast(
                filenames=cfg.data_cfg.val_paths,
                fields=cfg.data_cfg.fields,
                norm=cfg.data_cfg.normalize,
                time_window=cfg.data_cfg.time_window,
            )
    val_dataset.normalize(*normalization_constants)
    diff_term = normalization_constants[0].tolist()
    div_term = normalization_constants[1].tolist()
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    train_module = ForecastModule(
                model_cfg=cfg.model_cfg,
                data_cfg=cfg.data_cfg,
                optim_cfg=cfg.optim_cfg,
                scheduler_cfg=cfg.scheduler_cfg,
                log_wandb=cfg.use_wandb,
                normalization_constants=(diff_term, div_term),
            )

    trainer = Trainer(
        accelerator="gpu",
        devices=cfg.devices,
        strategy="ddp",
        max_epochs=cfg.max_epochs,
        logger=logger,
        default_root_dir=params["log_dir"],
    )

    if cfg.checkpoint_path:
        trainer.fit(
            train_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.checkpoint_path
        )
    else:
        trainer.fit(
            train_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
