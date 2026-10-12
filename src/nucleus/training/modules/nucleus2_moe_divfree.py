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
from nucleus.physics.div_projection import div_projection
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
    debed_diagnostics_every_n_steps: int = 100


@dataclass
class AuxiliaryTargets:
    """The target velocity split into its divergence-free and dilational parts, and
    its divergence, all normalized like the model's outputs."""
    velx_sol: torch.Tensor
    vely_sol: torch.Tensor
    velx_dil: torch.Tensor
    vely_dil: torch.Tensor
    div_source: torch.Tensor


def auxiliary_targets(target: FlashXBatch, grid: CellGrid) -> AuxiliaryTargets:
    # the velocities are normalized by a pure scale, so the linear projection and
    # divergence of the normalized velocity are the normalized physical ones
    velx, vely = target.fields["velfacex"], target.fields["velfacey"]
    velx_sol, vely_sol = div_projection(velx, vely, grid.dx, grid.dy)
    return AuxiliaryTargets(
        velx_sol=velx_sol,
        vely_sol=vely_sol,
        velx_dil=velx - velx_sol,
        vely_dil=vely - vely_sol,
        div_source=divergence_centers_from_faces(velx, vely, grid.dx, grid.dy),
    )


class Nucleus2MoEDivFreeTraining(nn.Module):
    """Trains a Nucleus2MoEDivFree on raw FlashXForecastBatches: flips the raw batch,
    normalizes it, adds noise to the normalized input, optionally pushes the input
    forward through the model, then takes the L1 loss over every predicted field and
    velocity part, an sdf sign loss, and the router loss."""

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
        data_loss, sdf_sign_loss = self._losses(pairs)
        loss = data_loss + sdf_sign_loss + router_loss(
            output.moe_outputs, self.settings.load_balance_loss_weight, self.settings.z_loss_weight
        )
        update_router_bias(self.model, output.moe_outputs)
        metrics = {
            "train/loss": loss, "train/data_loss": data_loss, "train/sdf_sign_loss": sdf_sign_loss,
            "train/num_unrolls": num_unrolls,
        }
        metrics |= _per_field_mae(pairs, "train") | moe_metrics(output.moe_outputs, "train")
        return StepOutput(loss, metrics)

    def validation_step(self, batch: FlashXForecastBatch) -> Metrics:
        batch = normalize_forecast_batch(self.model.normalizer, batch)
        target = batch.target.head_time_window(self.settings.target_timesteps)
        output = self.model(batch.input)
        pairs = self._prediction_target_pairs(output, target)
        data_loss, sdf_sign_loss = self._losses(pairs)
        metrics = {"val/loss": data_loss + sdf_sign_loss, "val/data_loss": data_loss, "val/sdf_sign_loss": sdf_sign_loss}
        return metrics | _per_field_mae(pairs, "val") | moe_metrics(output.moe_outputs, "val")

    @torch.no_grad()
    def gradient_metrics(self, step: int) -> Metrics:
        if step % self.settings.debed_diagnostics_every_n_steps != 0:
            return {}
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

    def _losses(self, pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """The mean absolute error over the cells of every predicted grid together,
        and the sdf sign loss."""
        total_abs_error = sum((prediction - goal).abs().sum() for prediction, goal in pairs.values())
        total_elements = sum(prediction.numel() for prediction, _ in pairs.values())
        return total_abs_error / total_elements, self._sdf_sign_loss(*pairs["dfun"])

    def _prediction_target_pairs(
        self, output: Nucleus2MoEDivFreeOutput, target: FlashXBatch
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        prediction = output.prediction.tail_time_window(target.num_timesteps)
        window = slice(-target.num_timesteps, None)
        auxiliary = auxiliary_targets(target, output.grid)
        pairs = {name: (prediction.fields[name], target.fields[name]) for name in self.model.field_names}
        for name in ("velx_sol", "vely_sol", "velx_dil", "vely_dil", "div_source"):
            pairs[name] = (getattr(output, name)[:, window], getattr(auxiliary, name))
        return pairs

    def _sdf_sign_loss(self, predicted_sdf: torch.Tensor, target_sdf: torch.Tensor) -> torch.Tensor:
        # penalize incorrect signs in the output SDF. Near the interface, an incorrect sign may result
        # in a small L1 error, and not be learned well. However, predicing phase incorrectly is extremely
        # problematic and so it need to be penalized heavily.
        predicted_physical = self.model.normalizer.unnormalize_field("dfun", predicted_sdf)
        target_physical = self.model.normalizer.unnormalize_field("dfun", target_sdf)
        num_vapor = (target_physical > 0).sum()
        num_liquid = target_physical.numel() - num_vapor
        vapor_weight = (num_liquid / num_vapor).item() if num_vapor > 0 else 1.0
        return sdf_sign_bce_loss(predicted_physical, target_physical, vapor_weight)


def _per_field_mae(pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]], prefix: str) -> Dict[str, torch.Tensor]:
    return {f"{prefix}/mae_{name}": (prediction - goal).abs().mean().detach() for name, (prediction, goal) in pairs.items()}
