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
from nucleus.physics.coordinates import CellGrid
from nucleus.physics.poisson import divergence_centers_from_faces
from nucleus.training.loop import Metrics, StepOutput
from nucleus.training.modules.common import normalize_forecast_batch
from nucleus.training.moe import moe_metrics, router_loss, update_router_bias
from nucleus.training.push_forward import push_forward, sample_num_unrolls
from nucleus.utils.losses import sdf_sign_bce_loss


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
    # add a cross entropy on the phase the sdf's sign gives, which a small L1 error
    # near the interface does not penalize
    use_sdf_sign_loss: bool = False
    # small, so the cross entropy, about 2 at the start, does not dominate the data loss
    sdf_sign_loss_weight: float = 0.01


def target_div_source(target: FlashXBatch, grid: CellGrid) -> torch.Tensor:
    """The divergence of the target velocity, which supervises the divergence source.
    The velocities are normalized by a pure scale, so the divergence of the
    normalized velocity is the normalized physical one."""
    return divergence_centers_from_faces(target.fields["velfacex"], target.fields["velfacey"], grid.dx, grid.dy)


class Nucleus2MoEDivFreeTraining(nn.Module):
    """Trains a Nucleus2MoEDivFree on raw FlashXForecastBatches: flips the raw batch,
    normalizes it, adds noise to the normalized input, optionally pushes the input
    forward through the model, then takes the L1 loss over every predicted field and
    the divergence source, optionally an sdf sign loss, and the router loss. The predicted face velocities are
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
        field_loss, field_metrics = self._field_loss(pairs, "train")
        loss = field_loss + router_loss(
            output.moe_outputs, self.settings.load_balance_loss_weight, self.settings.z_loss_weight
        )
        update_router_bias(self.model, output.moe_outputs)
        metrics = {"train/loss": loss, "train/num_unrolls": num_unrolls} | field_metrics
        metrics |= _per_field_mae(pairs, "train") | moe_metrics(output.moe_outputs, "train")
        return StepOutput(loss, metrics)

    def validation_step(self, batch: FlashXForecastBatch) -> Metrics:
        batch = normalize_forecast_batch(self.model.normalizer, batch)
        target = batch.target.head_time_window(self.settings.target_timesteps)
        output = self.model(batch.input)
        pairs = self._prediction_target_pairs(output, target)
        field_loss, field_metrics = self._field_loss(pairs, "val")
        metrics = {"val/loss": field_loss} | field_metrics
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

    def _field_loss(self, pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]], prefix: str) -> Tuple[torch.Tensor, Metrics]:
        """The data loss, plus the sdf sign loss when it is on, and their metrics."""
        data_loss = _data_loss(pairs)
        metrics = {f"{prefix}/data_loss": data_loss}
        if not self.settings.use_sdf_sign_loss:
            return data_loss, metrics
        sdf_sign_loss = self._sdf_sign_loss(*pairs["dfun"])
        total = data_loss + self.settings.sdf_sign_loss_weight * sdf_sign_loss
        return total, metrics | {f"{prefix}/sdf_sign_loss": sdf_sign_loss}

    def _sdf_sign_loss(self, predicted_sdf: torch.Tensor, target_sdf: torch.Tensor) -> torch.Tensor:
        # Near the interface a wrong sign is a small L1 error, but a wrong phase, so
        # the raw sdf is treated as the logit of the vapor phase. Vapor is weighted by
        # the liquid-to-vapor cell ratio, as it covers a small part of the domain.
        predicted_physical = self.model.normalizer.unnormalize_field("dfun", predicted_sdf)
        target_physical = self.model.normalizer.unnormalize_field("dfun", target_sdf)
        num_vapor = (target_physical > 0).sum()
        num_liquid = target_physical.numel() - num_vapor
        vapor_weight = (num_liquid / num_vapor).item() if num_vapor > 0 else 1.0
        return sdf_sign_bce_loss(predicted_physical, target_physical, vapor_weight)

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
