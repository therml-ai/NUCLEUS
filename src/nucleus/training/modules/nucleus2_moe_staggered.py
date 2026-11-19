from dataclasses import dataclass

from torch import nn
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.tasks.forecast.flashx_augmentation import (
    FlipAugmentation,
    ForecastAugmentation,
    GaussianNoiseAugmentation,
)
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch

from nucleus.models.nucleus2_moe_staggered import Nucleus2MoEStaggered
from nucleus.training.field_dropout import FieldDropout
from nucleus.training.loop import Metrics, StepOutput
from nucleus.training.modules.common import (
    PredictionTargetPairs,
    normalize_forecast_batch,
    per_field_mae,
    pooled_l1_loss,
)
from nucleus.training.moe import moe_metrics, router_loss, update_router_bias
from nucleus.training.push_forward import push_forward, sample_num_unrolls


@dataclass(frozen=True)
class Nucleus2MoEStaggeredTrainingSettings:
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


class Nucleus2MoEStaggeredTraining(nn.Module):
    """Trains a Nucleus2MoEStaggered as Nucleus2MoETraining trains a Nucleus2MoE:
    flips the raw batch, normalizes it, drops fields and adds noise to the
    normalized input, optionally pushes the input forward through the model, then
    takes the L1 loss over every predicted field on its own grid, plus the router
    loss."""

    def __init__(self, model: Nucleus2MoEStaggered, settings: Nucleus2MoEStaggeredTrainingSettings):
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
        pairs = self._prediction_target_pairs(output.prediction, batch.target)
        data_loss = pooled_l1_loss(pairs)
        loss = data_loss + router_loss(
            output.moe_outputs, self.settings.load_balance_loss_weight, self.settings.z_loss_weight
        )
        update_router_bias(self.model, output.moe_outputs)
        metrics = {"train/loss": loss, "train/data_loss": data_loss, "train/num_unrolls": num_unrolls}
        metrics |= per_field_mae(pairs, "train") | moe_metrics(output.moe_outputs, "train")
        return StepOutput(loss, metrics)

    def validation_step(self, batch: FlashXForecastBatch) -> Metrics:
        batch = normalize_forecast_batch(self.model.normalizer, batch)
        target = batch.target.head_time_window(self.settings.target_timesteps)
        output = self.model(batch.input)
        pairs = self._prediction_target_pairs(output.prediction, target)
        data_loss = pooled_l1_loss(pairs)
        metrics = {"val/loss": data_loss, "val/data_loss": data_loss}
        return metrics | per_field_mae(pairs, "val") | moe_metrics(output.moe_outputs, "val")

    def gradient_metrics(self) -> Metrics:
        return {}

    def _next_window(self, window: FlashXBatch) -> FlashXBatch:
        return self.model(window).prediction.tail_time_window(self.settings.target_timesteps)

    def _prediction_target_pairs(self, prediction: FlashXBatch, target: FlashXBatch) -> PredictionTargetPairs:
        prediction = prediction.tail_time_window(target.num_timesteps)
        return {name: (prediction.fields[name], target.fields[name]) for name in self.model.field_names}
