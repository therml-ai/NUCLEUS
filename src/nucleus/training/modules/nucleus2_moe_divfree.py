from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.tasks.forecast.flashx_augmentation import (
    FlipAugmentation,
    ForecastAugmentation,
    GaussianNoiseAugmentation,
)
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch

from nucleus.models.nucleus2_moe_divfree import Nucleus2MoEDivFree, Nucleus2MoEDivFreeOutput
from nucleus.physics.coordinates import CellCenteredGrid
from nucleus.physics.poisson import divergence_centers_from_faces
from nucleus.training.loop import Metrics, StepOutput
from nucleus.training.modules.common import normalize_forecast_batch
from nucleus.training.moe import moe_metrics, router_loss, update_router_bias
from nucleus.training.push_forward import push_forward, sample_num_unrolls


@dataclass(frozen=True)
class Nucleus2MoEDivFreeTrainingSettings:
    target_timesteps: int
    load_balance_loss_weight: float = 1e-5
    z_loss_weight: float = 1e-5
    flip_probability: float = 0.5
    min_noise_std: float = 1e-3
    max_noise_std: float = 5e-2
    # unrolls past the first target window; the dataset's unroll_steps minus one
    max_unrolls: int = 0
    push_forward_probability: float = 0.0


def target_div_source(target: FlashXBatch, grid: CellCenteredGrid) -> torch.Tensor:
    """The divergence of the target velocity, which supervises the divergence source.
    The velocities are normalized by a pure scale, so the divergence of the
    normalized velocity is the normalized physical one."""
    return divergence_centers_from_faces(target.fields["velfacex"], target.fields["velfacey"], grid.dx, grid.dy)


class Nucleus2MoEDivFreeTraining(nn.Module):
    """Trains a Nucleus2MoEDivFree on raw FlashXForecastBatches: flips the raw batch,
    normalizes it, adds noise to the normalized input, optionally pushes the input
    forward through the model, then takes the L1 loss over every predicted field and
    the divergence source, and the router loss. The predicted face velocities are
    the network's own unless the model's config sets use_div_projection."""

    def __init__(self, model: Nucleus2MoEDivFree, settings: Nucleus2MoEDivFreeTrainingSettings):
        super().__init__()
        self.model = model
        self.settings = settings
        # flips negate velocities and mirror the heater about the domain bounds,
        # which only holds for raw values
        self.raw_augmentation = ForecastAugmentation([FlipAugmentation(settings.flip_probability)])
        self.normalized_augmentation = ForecastAugmentation([
            GaussianNoiseAugmentation(settings.min_noise_std, settings.max_noise_std),
        ])

    def training_step(self, batch: FlashXForecastBatch, step: int) -> StepOutput:
        batch = self.normalized_augmentation(normalize_forecast_batch(self.model.normalizer, self.raw_augmentation(batch)))
        num_unrolls = sample_num_unrolls(self.settings.max_unrolls, self.settings.push_forward_probability)
        batch = push_forward(self._next_window, batch, num_unrolls, self.settings.target_timesteps)
        output = self.model(batch.input)
        pairs = self._prediction_target_pairs(output, batch.target)
        data_loss = _data_loss(pairs)
        loss = data_loss + router_loss(
            output.moe_outputs, self.settings.load_balance_loss_weight, self.settings.z_loss_weight
        )
        update_router_bias(self.model, output.moe_outputs)
        metrics = {"train/loss": loss, "train/data_loss": data_loss, "train/num_unrolls": num_unrolls}
        metrics |= _per_field_mae(pairs, "train") | moe_metrics(output.moe_outputs, "train")
        return StepOutput(loss, metrics)

    def validation_step(self, batch: FlashXForecastBatch) -> Metrics:
        batch = normalize_forecast_batch(self.model.normalizer, batch)
        target = batch.target.head_time_window(self.settings.target_timesteps)
        output = self.model(batch.input)
        pairs = self._prediction_target_pairs(output, target)
        data_loss = _data_loss(pairs)
        metrics = {"val/loss": data_loss, "val/data_loss": data_loss}
        return metrics | _per_field_mae(pairs, "val") | moe_metrics(output.moe_outputs, "val")

    @torch.no_grad()
    def gradient_metrics(self) -> Metrics:
        head = self.model.debed
        grads = torch.cat([param.grad.flatten() for param in head.parameters() if param.grad is not None])
        weights = torch.cat([param.flatten() for param in head.parameters() if param.ndim >= 2])
        return {
            "train/debed/grad_norm": grads.norm(),
            "train/debed/grad_absmax": grads.abs().max(),
            "train/debed/weight_std": weights.std(),
            "train/debed/weight_absmax": weights.abs().max(),
        }

    def _next_window(self, window: FlashXBatch) -> FlashXBatch:
        return self.model(window).prediction.tail_time_window(self.settings.target_timesteps)

    def _prediction_target_pairs(
        self, output: Nucleus2MoEDivFreeOutput, target: FlashXBatch
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        prediction = output.prediction.tail_time_window(target.num_timesteps)
        window = slice(-target.num_timesteps, None)
        pairs = {name: (prediction.fields[name], target.fields[name]) for name in self.model.field_names}
        pairs["div_source"] = (output.div_source[:, window], target_div_source(target, output.grid))
        return pairs

def _data_loss(pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """The mean absolute error over the cells of every predicted grid together."""
    total_abs_error = sum((prediction - goal).abs().sum() for prediction, goal in pairs.values())
    total_elements = sum(prediction.numel() for prediction, _ in pairs.values())
    return total_abs_error / total_elements


def _per_field_mae(pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]], prefix: str) -> Dict[str, torch.Tensor]:
    return {f"{prefix}/mae_{name}": (prediction - goal).abs().mean().detach() for name, (prediction, goal) in pairs.items()}
