import math
import signal
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Protocol

import torch
from torch import nn
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch

from nucleus.training.checkpoint import CheckpointManager, TrainingState
from nucleus.training.logger import Metric, MetricLogger
from nucleus.training.optim import Optimization

Metrics = Dict[str, Metric]
VALIDATION_LOSS = "val/loss"


@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: Metrics


class TrainingModule(Protocol):
    """What fit needs from a model's training module: an nn.Module holding the model
    that computes its own loss on a raw (unnormalized) batch. validation_step's
    metrics must include ``val/loss``."""
    model: nn.Module

    def train(self, mode: bool = True) -> nn.Module: ...

    def training_step(self, batch: FlashXForecastBatch, step: int) -> StepOutput: ...

    def validation_step(self, batch: FlashXForecastBatch) -> Metrics: ...

    def gradient_metrics(self) -> Metrics:
        """Metrics of the accumulated gradients, read just before the optimizer step
        of a step whose metrics are logged."""
        ...


@dataclass(frozen=True)
class LoopSettings:
    max_steps: int
    accumulate_grad_batches: int = 1
    val_every_n_steps: int = 1000
    log_every_n_steps: int = 50
    max_val_batches: Optional[int] = None


class NonFiniteValidationError(FloatingPointError):
    """The validation loss is NaN or infinite: training has diverged."""


class PreemptionSignal:
    """Records SIGTERM, which SLURM sends before preempting a job, so training can
    checkpoint at the end of the step rather than die in the middle of one."""

    def __init__(self):
        self.received = False

    def __enter__(self) -> "PreemptionSignal":
        self._previous_handler = signal.signal(signal.SIGTERM, self._handle)
        return self

    def __exit__(self, *exception) -> None:
        signal.signal(signal.SIGTERM, self._previous_handler)

    def _handle(self, signum, frame) -> None:
        self.received = True


def fit(
    module: TrainingModule,
    train_batches: Iterable[FlashXForecastBatch],
    val_batches: Optional[Iterable[FlashXForecastBatch]],
    optimization: Optimization,
    settings: LoopSettings,
    logger: MetricLogger,
    device: torch.device,
    checkpoints: Optional[CheckpointManager] = None,
    state: Optional[TrainingState] = None,
) -> TrainingState:
    """Trains until state.step reaches settings.max_steps, counting optimizer steps,
    so a run resumed from a checkpoint's state continues where it stopped."""
    state = state or TrainingState()
    batches = _endless(train_batches)
    with PreemptionSignal() as preemption:
        while state.step < settings.max_steps and not preemption.received:
            logged = (state.step + 1) % settings.log_every_n_steps == 0
            metrics = _optimizer_step(module, batches, optimization, settings, device, state.step, logged)
            state.step += 1
            if logged:
                logger.log(metrics, state.step)
            if val_batches is not None and state.step % settings.val_every_n_steps == 0:
                _validate_and_checkpoint(module, val_batches, settings, logger, device, checkpoints, state)
    if checkpoints is not None:
        checkpoints.save_last(state)
    return state


def _optimizer_step(
    module: TrainingModule,
    batches: Iterator[FlashXForecastBatch],
    optimization: Optimization,
    settings: LoopSettings,
    device: torch.device,
    step: int,
    logged: bool,
) -> Metrics:
    """The step's metrics; the gradient ones only on steps whose metrics are logged,
    as reading the gradients costs a pass over every parameter."""
    module.train()
    started = time.perf_counter()
    for _ in range(settings.accumulate_grad_batches):
        output = module.training_step(next(batches).to(device, non_blocking=True), step)
        (output.loss / settings.accumulate_grad_batches).backward()
    metrics = dict(output.metrics)
    if logged:
        metrics |= module.gradient_metrics()
        metrics["train/grad_norm"] = torch.nn.utils.clip_grad_norm_(module.model.parameters(), max_norm=float("inf"))
    metrics["train/learning_rate"] = optimization.learning_rate()
    optimization.step()
    metrics["train/iterations_per_second"] = 1.0 / (time.perf_counter() - started)
    return metrics


def _validate_and_checkpoint(
    module: TrainingModule,
    val_batches: Iterable[FlashXForecastBatch],
    settings: LoopSettings,
    logger: MetricLogger,
    device: torch.device,
    checkpoints: Optional[CheckpointManager],
    state: TrainingState,
) -> None:
    metrics = validate(module, val_batches, device, settings.max_val_batches)
    logger.log(metrics, state.step)
    # raised before checkpointing, so the diverged model replaces neither checkpoint
    _require_finite_validation_loss(metrics, state.step)
    if checkpoints is not None:
        checkpoints.save_if_best(metrics[VALIDATION_LOSS], state)
        checkpoints.save_last(state)


def _require_finite_validation_loss(metrics: Dict[str, float], step: int) -> None:
    if math.isfinite(metrics[VALIDATION_LOSS]):
        return
    non_finite = {name: value for name, value in sorted(metrics.items()) if not math.isfinite(value)}
    raise NonFiniteValidationError(
        f"validation at step {step} gave {VALIDATION_LOSS} = {metrics[VALIDATION_LOSS]}, so training "
        f"has diverged and stops; the non-finite validation metrics were {non_finite}"
    )


@torch.no_grad()
def validate(
    module: TrainingModule,
    val_batches: Iterable[FlashXForecastBatch],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """The mean of every validation metric over the batches."""
    module.eval()
    totals: Dict[str, float] = {}
    num_batches = 0
    for batch in val_batches:
        if max_batches is not None and num_batches >= max_batches:
            break
        for name, value in module.validation_step(batch.to(device, non_blocking=True)).items():
            totals[name] = totals.get(name, 0.0) + float(value)
        num_batches += 1
    if num_batches == 0:
        raise ValueError("there were no validation batches")
    return {name: total / num_batches for name, total in totals.items()}


def _endless(batches: Iterable[FlashXForecastBatch]) -> Iterator[FlashXForecastBatch]:
    while True:
        yield from batches
