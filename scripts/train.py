import os
import pprint
import time
import signal
from datetime import date
import subprocess
import glob
from pathlib import Path
from typing import Optional

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelSummary, Callback, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.plugins.environments import SLURMEnvironment

from nucleus.data.batching import collate
from nucleus.data.normalize import get_normalizer
from nucleus.data import ForecastDataset, InMemForecastDataset
from nucleus.modules import get_train_module
from nucleus.utils.set_fp32_precision import set_fp32_precision
from nucleus.utils.parameter_count import count_model_parameters

def get_git_sha(directory: Path) -> Optional[str]:
    print(directory)
    # Base case: if we reach the root directory, there's no .git directory.
    # If this happens, there's something wrong with the directory structure.
    if directory == Path("/"):
        print(f"Reached root directory, without finding .git directory.")
        return None
    contains_dot_git_dir = (directory / ".git").exists()
    if contains_dot_git_dir:
        git_sha = (directory / ".git" / "refs" / "heads" / "main").read_text().strip()
        return git_sha
    return get_git_sha(directory.parent)

def is_leader_process():
    """
    Check if the current process is the leader process.
    """
    if os.getenv("SLURM_PROCID") is None:
        if os.getenv("LOCAL_RANK") is not None:
            return int(os.getenv("LOCAL_RANK")) == 0
        else:
            return True
    else:
        return os.getenv("SLURM_PROCID") == "0"

class PreemptionCheckpointCallback(Callback):
    """
    Tries to save a checkpoint when a SIGTERM signal is received.
    Args:
        checkpoint_path: Path to save the checkpoint.
    """
    def __init__(self, checkpoint_path="preemption_checkpoint.ckpt"):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.already_handled = False

    def setup(self, trainer, pl_module, stage: str) -> None:
        self.trainer = trainer
        # Register the signal handler for SIGTERM in case of job preemption due to paid job
        signal.signal(signal.SIGTERM, self.handle_preemption)

    def handle_preemption(self, signum, frame):
        """
        Handle the SIGTERM signal.
        """
        if self.already_handled:
            return
        self.already_handled = True
        try:
            # Save the checkpoint. Use trainer.save_checkpoint if accessible.
            # Note: You might need to call this on the main thread.
            self.trainer.save_checkpoint(self.checkpoint_path)
            print(f"Due to preemption Checkpoint saved to {self.checkpoint_path}.")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        # delay a bit to ensure the checkpoint save finishes.
        time.sleep(5)

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    set_fp32_precision()
    
    # Setup Wandb Logger.
    log_id_parts = [
        cfg.model_cfg.name.lower(),
        cfg.data_cfg.dataset.lower(),
        date.today().strftime("%Y-%m-%d"),
    ]
    if os.getenv("SLURM_JOB_ID") is not None:
        log_id_parts.append(os.getenv("SLURM_JOB_ID"))
    
    log_id = "_".join(log_id_parts)
    cfg.log_dir = os.path.join(cfg.log_dir, log_id)
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    commit_sha = get_git_sha(Path.cwd())
    if commit_sha is None:
        print("Failed to get commit SHA. Saving in config as None.")
    cfg.commit_sha = commit_sha

    logger = WandbLogger(
        entity="hpcforge",
        project="bubbleformer",
        name=log_id,
        dir=cfg.log_dir,
        config=OmegaConf.to_container(cfg),
    )

    dataset = InMemForecastDataset if "64" in cfg.data_cfg.dataset else ForecastDataset
    
    normalizer = get_normalizer(OmegaConf.to_container(cfg.normalizer_cfg, resolve=True))

    train_dataset = dataset(
        filenames=cfg.data_cfg.train_paths,
        input_fields=cfg.data_cfg.input_fields,
        output_fields=cfg.data_cfg.output_fields,
        history_time_window=cfg.history_time_window,
        future_time_window=cfg.future_time_window,
        time_step=cfg.time_step,
        start_time=cfg.start_time,
        normalizer=normalizer,
        augment=True,
        layout=cfg.model_cfg.layout
    )
    val_dataset = dataset(
        filenames=cfg.data_cfg.val_paths,
        input_fields=cfg.data_cfg.input_fields,
        output_fields=cfg.data_cfg.output_fields,
        history_time_window=cfg.history_time_window,
        future_time_window=cfg.future_time_window,
        time_step=cfg.time_step,
        start_time=cfg.start_time,
        normalizer=normalizer,
        augment=False,
        layout=cfg.model_cfg.layout
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate,
    )
    
    train_module = get_train_module(cfg.model_cfg.train_module_name)(
        checkpoint_path=cfg.checkpoint_path,
        model_cfg=cfg.model_cfg,
        data_cfg=cfg.data_cfg,
        normalizer_cfg=cfg.normalizer_cfg,
        optim_cfg=cfg.optim_cfg,
        scheduler_cfg=cfg.scheduler_cfg,
        log_wandb=False,
    )

    active_params = count_model_parameters(train_module.model, active=True)
    total_params = count_model_parameters(train_module.model, active=False)
    print(f"Active Model parameters: {active_params:,d}")
    print(f"Total Model parameters: {total_params:,d}")

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        )
    )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.devices,
        num_nodes=cfg.nodes,
        strategy="auto",
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        val_check_interval=cfg.val_check_interval,
        log_every_n_steps=100,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        logger=logger,
        default_root_dir=cfg.log_dir,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
        enable_model_summary=True,
        num_sanity_val_steps=0,
        callbacks=[
            ModelSummary(max_depth=-1), 
            ModelCheckpoint(
                dirpath=cfg.log_dir + "/checkpoints",
                monitor="val/loss",
                mode="min",
                save_top_k=2,
                save_last=True,
                every_n_train_steps=20000,
                save_on_exception=True
            ),
            progress_bar
        ],
    )
    
    if is_leader_process():
        pp = pprint.PrettyPrinter(depth=4)
        pp.pprint(cfg)

    #torch.cuda.memory._record_memory_history(
    #    max_entries=100000
    #)

    #with profile(
    #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #record_shapes=True,
        #profile_memory=True,
    #) as prof:

    trainer.fit(
        train_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    #prof.export_memory_timeline("memory_timeline.html", device="cuda:0")
    #prof.export_chrome_trace("trace.json")
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    #rint(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    #try:
    #    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    #except Exception as e:
    #    print("failed to capture memory snapshot")
    #torch.cuda.memory._record_memory_history(
    #    enabled=None
    #)

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
