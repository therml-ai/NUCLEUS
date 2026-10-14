import os
import random
import subprocess
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from boiling_data.flashx.normalizer import NormalizerWrapper

from nucleus.models import get_model, get_model_class
from nucleus.testing.reference_attention import neighborhood_attention_for
from nucleus.training.checkpoint import CheckpointManager, TrainingState, restore_training
from nucleus.training.data import ForecastDataSettings, forecast_dataloader, read_split
from nucleus.training.logger import MetricLogger
from nucleus.training.loop import LoopSettings, TrainingModule, fit
from nucleus.training.optim import build_optimization
from nucleus.utils.parameter_count import count_model_parameters

# a smoke run checks the training loop end to end in seconds and writes nothing
SMOKE_STEPS = 3


def run_training(cfg: DictConfig, build_training_module: Callable[[nn.Module, DictConfig], TrainingModule]) -> TrainingState:
    """Trains the model named by cfg.model_cfg with the training module that
    build_training_module wraps it in, as a model's train script configures."""
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    module = build_training_module(model, cfg)
    print(f"active parameters: {count_model_parameters(model, active=True):,d}")
    print(f"total parameters: {count_model_parameters(model, active=False):,d}")

    data_settings = _data_settings(cfg)
    split = read_split(cfg.split_dir)
    train_batches = forecast_dataloader(split["train"], model.field_names, model.config_keys, data_settings, shuffle=True)
    val_batches = forecast_dataloader(split["val"], model.field_names, model.config_keys, data_settings, shuffle=False)
    loop_settings = _loop_settings(cfg)
    optimization = build_optimization(
        model, _container(cfg.optim_cfg), _container(cfg.scheduler_cfg), loop_settings.max_steps
    )
    state = restore_training(cfg.checkpoint_path, model, optimization, device) if cfg.checkpoint_path else None

    logger, checkpoints = MetricLogger(), None
    if not cfg.smoke:
        run_directory = make_run_directory(cfg)
        logger = _wandb_logger(cfg, run_directory)
        checkpoints = CheckpointManager(run_directory / "checkpoints", model, optimization, _container(cfg))
    try:
        with neighborhood_attention_for(device):
            state = fit(module, train_batches, val_batches, optimization, loop_settings, logger, device, checkpoints, state)
    except BaseException:
        logger.finish(failed=True)
        raise
    logger.finish()
    return state


def build_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    """The model with a normalizer read from the data config's statistics file."""
    model_class = get_model_class(cfg.model_cfg.name)
    normalizer = NormalizerWrapper.from_json(
        cfg.data_cfg.statistics_path, vector_fields=getattr(model_class, "vector_fields", ())
    )
    model = get_model(cfg.model_cfg.name, normalizer, **_container(cfg.model_cfg.params)).to(device)
    if cfg.compile and device.type == "cuda":
        # only the tensor step: the batch-level forward is not traceable with fullgraph
        model.step = torch.compile(model.step, fullgraph=True)
    return model


def make_run_directory(cfg: DictConfig) -> Path:
    parts = [cfg.model_cfg.name, date.today().isoformat()]
    if os.getenv("SLURM_JOB_ID") is not None:
        parts.append(os.environ["SLURM_JOB_ID"])
    run_directory = Path(cfg.log_dir) / "_".join(parts)
    run_directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_directory / "config.yaml")
    return run_directory


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _wandb_logger(cfg: DictConfig, run_directory: Path) -> MetricLogger:
    if not cfg.wandb.enabled:
        return MetricLogger()
    config = _container(cfg) | {"commit": git_commit()}
    return MetricLogger.wandb(cfg.wandb.project, cfg.wandb.entity, run_directory.name, str(run_directory), config)


def _data_settings(cfg: DictConfig) -> ForecastDataSettings:
    settings = _container(cfg.data)
    if cfg.smoke:
        # errors in worker processes are harder to read
        settings["num_workers"] = 0
    return ForecastDataSettings(**settings)


def _loop_settings(cfg: DictConfig) -> LoopSettings:
    settings = _container(cfg.loop)
    if cfg.smoke:
        settings |= {
            "max_steps": SMOKE_STEPS, "val_every_n_steps": SMOKE_STEPS, "log_every_n_steps": 1, "max_val_batches": 1,
        }
    return LoopSettings(**settings)


def _container(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)
