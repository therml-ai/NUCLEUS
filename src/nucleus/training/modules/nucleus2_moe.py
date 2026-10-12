from dataclasses import dataclass

import torch
from torch import nn
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.tasks.forecast.flashx_augmentation import (
    FlipAugmentation,
    ForecastAugmentation,
    GaussianNoiseAugmentation,
)
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch

from nucleus.models.nucleus2_moe import Nucleus2MoE
from nucleus.training.field_dropout import FieldDropout
from nucleus.training.loop import Metrics, StepOutput
from nucleus.training.metrics import prediction_metrics
from nucleus.training.modules.common import normalize_forecast_batch
from nucleus.training.moe import moe_metrics, router_loss, update_router_bias
from nucleus.training.push_forward import push_forward, sample_num_unrolls


@dataclass(frozen=True)
class Nucleus2MoETrainingSettings:
    target_timesteps: int
    load_balance_loss_weight: float = 1e-5
    z_loss_weight: float = 1e-5
    flip_probability: float = 0.5
    field_drop_probability: float = 0.04
    min_noise_std: float = 1e-3
    max_noise_std: float = 5e-2
    # unrolls past the first target window; the dataset's unroll_steps minus one
    max_unrolls: int = 0
    push_forward_probability: float = 0.0


class Nucleus2MoETraining(nn.Module):
    """Trains a Nucleus2MoE on raw FlashXForecastBatches: flips the raw batch,
    normalizes it with the model's normalizer, drops fields and adds noise to the
    normalized input, optionally pushes the input forward through the model, then
    takes the L1 loss of the next target window plus the router loss."""

    def __init__(self, model: Nucleus2MoE, settings: Nucleus2MoETrainingSettings):
        super().__init__()
        self.model = model
        self.settings = settings
        # flips negate velocities and mirror the heater about the domain bounds,
        # which only holds for raw values
        self.raw_augmentation = ForecastAugmentation([FlipAugmentation(settings.flip_probability)])
        self.normalized_augmentation = ForecastAugmentation([
            FieldDropout(settings.field_drop_probability),
            GaussianNoiseAugmentation(settings.min_noise_std, settings.max_noise_std),
        ])

    def training_step(self, batch: FlashXForecastBatch, step: int) -> StepOutput:
        batch = self.normalized_augmentation(normalize_forecast_batch(self.model.normalizer, self.raw_augmentation(batch)))
        num_unrolls = sample_num_unrolls(self.settings.max_unrolls, self.settings.push_forward_probability)
        batch = push_forward(self._next_window, batch, num_unrolls, self.settings.target_timesteps)
        output = self.model(batch.input)
        prediction, target = self._stacked(output.prediction, batch.target)
        data_loss = torch.nn.functional.l1_loss(prediction, target)
        loss = data_loss + router_loss(
            output.moe_outputs, self.settings.load_balance_loss_weight, self.settings.z_loss_weight
        )
        update_router_bias(self.model, output.moe_outputs)
        metrics = {"train/loss": loss, "train/data_loss": data_loss, "train/num_unrolls": num_unrolls}
        metrics |= prediction_metrics(prediction, target, "train") | moe_metrics(output.moe_outputs, "train")
        return StepOutput(loss, metrics)

    def validation_step(self, batch: FlashXForecastBatch) -> Metrics:
        batch = normalize_forecast_batch(self.model.normalizer, batch)
        output = self.model(batch.input)
        prediction, target = self._stacked(
            output.prediction, batch.target.head_time_window(self.settings.target_timesteps)
        )
        metrics = {"val/loss": torch.nn.functional.l1_loss(prediction, target)}
        return metrics | prediction_metrics(prediction, target, "val") | moe_metrics(output.moe_outputs, "val")

    def gradient_metrics(self, step: int) -> Metrics:
        return {}

    def _next_window(self, window: FlashXBatch) -> FlashXBatch:
        return self.model(window).prediction.tail_time_window(self.settings.target_timesteps)

    def _stacked(self, prediction: FlashXBatch, target: FlashXBatch):
        names = self.model.field_names
        return (
            prediction.tail_time_window(target.num_timesteps).stacked_fields(names),
            target.stacked_fields(names),
        )
