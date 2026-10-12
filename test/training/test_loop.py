import math

import pytest
import torch

from nucleus.models import load_model
from nucleus.models.nucleus2_moe import Nucleus2MoE
from nucleus.testing.flashx import random_forecast_batch
from nucleus.training.checkpoint import BEST_CHECKPOINT, LAST_CHECKPOINT, CheckpointManager, restore_training
from nucleus.training.logger import MetricLogger
from nucleus.training.loop import LoopSettings, fit, validate
from nucleus.training.optim import build_optimization

OPTIM = {"name": "adamw", "params": {"lr": 1e-3, "weight_decay": 0.1}}
SCHEDULER = {"name": "cosine_warmup", "params": {"warmup": 1, "eta_min": 1e-6}}


def _batches(num_batches=2):
    return [
        random_forecast_batch(Nucleus2MoE.field_names, Nucleus2MoE.config_keys)
        for _ in range(num_batches)
    ]


def _settings(**overrides):
    return LoopSettings(**{"max_steps": 3, "val_every_n_steps": 2, "log_every_n_steps": 1} | overrides)


def test_fit_takes_max_steps_optimizer_steps_and_changes_the_weights(moe_training):
    before = [param.detach().clone() for param in moe_training.model.parameters()]
    optimization = build_optimization(moe_training.model, OPTIM, SCHEDULER, max_steps=3)
    state = fit(moe_training, _batches(), _batches(1), optimization, _settings(), MetricLogger(), torch.device("cpu"))
    assert state.step == 3
    # without a checkpoint manager no best validation loss is recorded
    assert state.best_val_loss == math.inf
    after = list(moe_training.model.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


def test_gradient_accumulation_draws_several_batches_per_step(moe_training):
    drawn = []

    class Recording(list):
        def __iter__(self):
            for batch in super().__iter__():
                drawn.append(batch)
                yield batch

    optimization = build_optimization(moe_training.model, OPTIM, SCHEDULER, max_steps=2)
    fit(
        moe_training, Recording(_batches(3)), None, optimization,
        _settings(max_steps=2, accumulate_grad_batches=3), MetricLogger(), torch.device("cpu"),
    )
    assert len(drawn) == 6


def test_checkpoints_resume_the_step_and_rebuild_the_model(moe_training, tmp_path):
    optimization = build_optimization(moe_training.model, OPTIM, SCHEDULER, max_steps=3)
    checkpoints = CheckpointManager(tmp_path, moe_training.model, optimization, config={"run": "test"})
    fit(moe_training, _batches(), _batches(1), optimization, _settings(), MetricLogger(), torch.device("cpu"), checkpoints)
    assert (tmp_path / LAST_CHECKPOINT).exists() and (tmp_path / BEST_CHECKPOINT).exists()

    state = restore_training(tmp_path / LAST_CHECKPOINT, moe_training.model, optimization)
    assert state.step == 3 and math.isfinite(state.best_val_loss)
    resumed = fit(
        moe_training, _batches(), None, optimization, _settings(max_steps=4), MetricLogger(), torch.device("cpu"),
        state=state,
    )
    assert resumed.step == 4

    loaded = load_model(tmp_path / LAST_CHECKPOINT)
    assert loaded.normalizer.statistics == moe_training.model.normalizer.statistics


def test_validate_averages_the_metrics_over_batches(moe_training):
    metrics = validate(moe_training, _batches(2), torch.device("cpu"))
    assert math.isfinite(metrics["val/loss"])
    with pytest.raises(ValueError, match="no validation batches"):
        validate(moe_training, [], torch.device("cpu"))


class _RecordingLogger(MetricLogger):
    def __init__(self):
        super().__init__()
        self.logged = {}

    def log(self, metrics, step):
        self.logged[step] = dict(metrics)


def test_every_logged_step_has_the_gradient_norm(moe_training):
    logger = _RecordingLogger()
    optimization = build_optimization(moe_training.model, OPTIM, SCHEDULER, max_steps=4)
    fit(
        moe_training, _batches(), None, optimization,
        _settings(max_steps=4, log_every_n_steps=2), logger, torch.device("cpu"),
    )
    assert sorted(logger.logged) == [2, 4]
    for metrics in logger.logged.values():
        assert math.isfinite(float(metrics["train/grad_norm"]))
        assert "train/loss" in metrics
