import os
import pprint
import time
import signal
from datetime import date
from pathlib import Path
from typing import Optional

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelSummary, Callback, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.plugins.environments import SLURMEnvironment

import braceexpand

from nucleus.data.batching import collate
from nucleus.data.normalize import get_normalizer
from nucleus.data import get_pydataset, forecast_web_dataset
from nucleus.models.modules import get_train_module
from nucleus.utils.set_fp32_precision import set_fp32_precision
from nucleus.utils.parameter_count import count_model_parameters


class ProfilerCallback(Callback):
    """
    Profiles a fixed number of training iterations and writes a Chrome trace.
    Uses wait=1 to skip the first step (often slow due to compilation),
    then warmup=1, then active=steps_to_profile active steps.
    """
    def __init__(self, steps_to_profile: int = 5, output_path: str = "profile_trace.json"):
        self.steps_to_profile = steps_to_profile
        self.output_path = output_path
        self._profiler = None

    def on_train_start(self, trainer, pl_module):
        self._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=self.steps_to_profile, repeat=1),
            record_shapes=True,
            with_stack=False,
        )
        self._profiler.__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._profiler.step()
        total_steps = 1 + 1 + self.steps_to_profile  # wait + warmup + active
        if batch_idx + 1 >= total_steps:
            self._profiler.__exit__(None, None, None)
            self._profiler.export_chrome_trace(self.output_path)
            self._profiler = None
            print(f"\nProfile trace written to: {self.output_path}")
            trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)

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
    
    run = wandb.init(
        project="nucleus",
        entity="hpcforge"
    )
    
    assert cfg.log_dir is not None, "Must set `log_dir` in hydra config"
    print(f"logging to {cfg.log_dir}")

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
        entity=run.entity,
        project=run.project,
        name=log_id,
        dir=cfg.log_dir,
        config=OmegaConf.to_container(cfg),
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
    
    normalizer = get_normalizer(OmegaConf.to_container(cfg.normalizer_cfg, resolve=True))

    collate_fn = collate
    shared_dataset_kwargs = dict(
        history_time_window=cfg.history_time_window,
        future_time_window=cfg.future_time_window,
        fluid_params=train_module.model.expected_fluid_params,
        heater_params=train_module.model.expected_heater_params,
        global_params=train_module.model.expected_global_params,
        layout=train_module.model.layout,
        normalizer=normalizer,
    )

    pydataset = cfg.pydataset
    use_webdataset = pydataset == "forecast_web"
    if use_webdataset:
        train_shard_urls = list(braceexpand.braceexpand(list(cfg.data_cfg.train_paths)[0]))

        train_dataset = forecast_web_dataset(
            shard_urls=train_shard_urls,
            cache_dir=None,
            cache_size=0,
            augment=True,
            **shared_dataset_kwargs,
        )
        val_dataset = forecast_web_dataset(
            shard_urls=list(cfg.data_cfg.val_paths)[0],
            cache_dir=None,
            cache_size=0,
            augment=False,
            **shared_dataset_kwargs,
        )
    else:
        dataset_cls = get_pydataset(pydataset)
        hdf5_kwargs = dict(
            time_step=cfg.time_step,
            start_time=cfg.start_time,
            input_fields=cfg.data_cfg.input_fields,
            output_fields=cfg.data_cfg.output_fields,
        )
        train_dataset = dataset_cls(
            filenames=cfg.data_cfg.train_paths,
            augment=True,
            **shared_dataset_kwargs,
            **hdf5_kwargs,
        )
        val_dataset = dataset_cls(
            filenames=cfg.data_cfg.val_paths,
            augment=False,
            **shared_dataset_kwargs,
            **hdf5_kwargs,
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=not use_webdataset,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=not use_webdataset,
        multiprocessing_context='fork',
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=not use_webdataset,
        multiprocessing_context='fork',
        collate_fn=collate_fn,
    )

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

    callbacks = [
        ModelSummary(max_depth=-1),
        ModelCheckpoint(
            dirpath=cfg.log_dir + "/checkpoints",
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last=True,
            #every_n_train_steps=20000,
            every_n_epochs=1,
            save_on_exception=True
        ),
        progress_bar,
    ]
    if cfg.get("profile", False):
        profile_path = os.path.join(cfg.log_dir, "profile_trace.json")
        callbacks.append(ProfilerCallback(steps_to_profile=5, output_path=profile_path))
        print(f"Profiling enabled. Trace will be written to: {profile_path}")

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.devices,
        num_nodes=cfg.nodes,
        strategy="auto",
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        val_check_interval=cfg.val_check_interval,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        log_every_n_steps=100,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        logger=logger,
        default_root_dir=cfg.log_dir,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
        enable_model_summary=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    
    if is_leader_process():
        pp = pprint.PrettyPrinter(depth=4)
        pp.pprint(cfg)

    trainer.fit(
        train_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
