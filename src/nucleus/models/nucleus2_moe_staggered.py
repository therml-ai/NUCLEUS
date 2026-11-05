"""Data-only NUCLEUS-2 on the staggered (MAC) grid: the backbone with a head that
predicts sdf, temperature and face velocities, trained on an L1 data loss."""
import math
from dataclasses import dataclass, replace
from typing import List, Optional

import torch

from nucleus.data.in_mem_divfree_forecast_dataset import DivFreeBatch, DivFreeData
from nucleus.layers.moe.topk_moe import TopkMoEOutput
from nucleus.models.modules import MoEConditionedForecastModule
from nucleus.noise import FieldDropout, LogUniformNoise
from nucleus.physics.poisson import divergence_centers_from_faces, GRID_SPACING
from nucleus.trajectory import Trajectory
from nucleus.utils.inf_stabilizer import clip_temp_by_phase
from nucleus.models.nucleus2_moe_staggered_backbone import (
    Nucleus2MoEStaggeredBackbone,
    Nucleus2MoEStaggeredBackboneConfig,
    StaggeredFields,
    config_from_dict,
    faces_from_cell_split,
    staggered_to_cells,
)

from ._api import register_model

__all__ = [
    "Nucleus2MoEStaggered",
    "Nucleus2MoEStaggeredBase",
    "Nucleus2MoEStaggeredConfig",
    "Nucleus2MoEStaggeredOutput",
    "StaggeredForecastModule",
    "faces_from_cell_split",
]

# Grid spacing used for the divergence diagnostic, in normalized units.
GRADIENT_SPACING = GRID_SPACING


@dataclass
class Nucleus2MoEStaggeredConfig(Nucleus2MoEStaggeredBackboneConfig):
    pass


def _config_from_dict(d: dict) -> Nucleus2MoEStaggeredConfig:
    return config_from_dict(Nucleus2MoEStaggeredConfig, d)


@dataclass
class Nucleus2MoEStaggeredOutput:
    """Model output, each field on its natural grid and in normalized units.

    Shapes: ``sdf``, ``temperature`` ``(B, T, H, W)``; ``velx`` ``(B, T, H, W+1)``;
    ``vely`` ``(B, T, H+1, W)``.
    """
    sdf: torch.Tensor
    temperature: torch.Tensor
    velx: torch.Tensor
    vely: torch.Tensor
    moe_outputs: List[TopkMoEOutput]

    def to_cell_tensor(self) -> torch.Tensor:
        """The same six cell channels the model consumes (for logging/plotting)."""
        return staggered_to_cells(self.sdf, self.temperature, self.velx, self.vely)


class Nucleus2MoEStaggeredBase(Nucleus2MoEStaggeredBackbone):
    """Uncompiled model. Use this class directly when forward hooks are needed
    (``torch.compile(fullgraph=True)`` does not run them); the registered
    ``nucleus2_moe_staggered`` is this class compiled, with identical weights."""

    config_class = Nucleus2MoEStaggeredConfig
    # Rollout code uses this to choose staggered (natural-grid) trajectories.
    staggered_io = True

    def __init__(self, config: Nucleus2MoEStaggeredConfig):
        # Output channels = input channels: sdf, temperature, and the four face copies.
        super().__init__(config, out_channels=len(self.expected_fields))

    def forward(self, batch: DivFreeBatch, normalizer=None) -> Nucleus2MoEStaggeredOutput:
        return self.step(batch)

    def step(self, input, sim_params: Optional[torch.Tensor] = None, normalizer=None, **unused):
        """One forecast window. ``input`` is a :class:`DivFreeBatch` (which carries the
        conditioning tensor) or any container with ``sdf``, ``temperature``, ``velx``,
        ``vely`` on their natural grids, with ``sim_params`` passed separately. Extra
        keyword arguments are accepted and ignored, so callers written for the divfree
        model's ``step`` signature also work."""
        if isinstance(input, DivFreeBatch):
            sim_params = input.sim_params_tensor
            input = input.input
        assert sim_params is not None and sim_params.dtype == torch.float32

        x, moe_outputs = self.model_step(input, sim_params)
        sdf, temperature, vel_left, vel_right, vel_bottom, vel_top = x.float().unbind(-1)
        velx, vely = faces_from_cell_split(vel_left, vel_right, vel_bottom, vel_top)
        return Nucleus2MoEStaggeredOutput(
            sdf=sdf, temperature=temperature, velx=velx, vely=vely, moe_outputs=moe_outputs,
        )

    def _normalized_sim_params(self, sim_params_dict: dict, normalizer, device, batch_size: int) -> torch.Tensor:
        normalized = normalizer.normalize_params([sim_params_dict])[0]
        values = (
            [normalized[param] for param in self.expected_fluid_params]
            + [normalized["heater"][param] for param in self.expected_heater_params]
            + [normalized[param] for param in self.expected_global_params]
        )
        sim_params = torch.tensor(values, device=device, dtype=torch.float32)
        return sim_params[None, :].expand(batch_size, -1).contiguous()

    def forward_trajectory(
        self,
        trajectory: Trajectory,
        normalizer,
        dx: float,
        input_time_window_size: int,
        output_time_window_size: int,
        trajectory_steps: int,
        return_moe_outputs: bool = False,
        clip_temp: bool = False,
    ):
        """Autoregressive rollout on the natural grids. ``trajectory`` and the result
        are in physical units; ``dx`` is accepted for a common signature."""
        assert input_time_window_size <= trajectory.num_steps
        sim_params_dict = trajectory.sim_params[0]
        bulk_temp = sim_params_dict["bulk_temp"]
        sim_params = self._normalized_sim_params(
            sim_params_dict, normalizer, trajectory.sdf.device, trajectory.sdf.shape[0]
        )
        keep = slice(-output_time_window_size, None)
        trajectory_moe_outputs = [] if return_moe_outputs else None

        for _ in range(input_time_window_size, trajectory_steps, output_time_window_size):
            window = trajectory.last(input_time_window_size)
            model_input = StaggeredFields(
                sdf=normalizer.normalize_sdf(window.sdf),
                temperature=normalizer.normalize_temp(window.temp, bulk_temp),
                velx=normalizer.normalize_velx(window.velx),
                vely=normalizer.normalize_vely(window.vely),
            )
            output = self.step(model_input, sim_params)
            sdf = normalizer.unnormalize_sdf(output.sdf)[:, keep]
            temp = normalizer.unnormalize_temp(output.temperature, bulk_temp)[:, keep]
            velx = normalizer.unnormalize_velx(output.velx)[:, keep]
            vely = normalizer.unnormalize_vely(output.vely)[:, keep]
            if clip_temp:
                temp = clip_temp_by_phase(
                    temp, sdf, sim_params_dict["sat_temp"], sim_params_dict["heater"]["wallTemp"],
                )
            trajectory = trajectory.extend(sdf, temp, velx, vely)
            if return_moe_outputs:
                trajectory_moe_outputs.append(output.moe_outputs)

        if return_moe_outputs:
            return trajectory, trajectory_moe_outputs
        return trajectory


@register_model("nucleus2_moe_staggered")
@torch.compile(fullgraph=True)
class Nucleus2MoEStaggered(Nucleus2MoEStaggeredBase):
    pass


class StaggeredForecastModule(MoEConditionedForecastModule):
    """L1 over sdf, temperature and face velocities plus the router terms.
    ``model_cfg.params.augmentation``: ``cell_centered`` (one noise scale per sample
    shared by all fields, then FieldDropout, as the cell-centered modules) or ``noise``
    (independent noise per field)."""

    LOSS_FIELDS = ("sdf", "temperature", "velx", "vely")
    INPUT_FIELDS = ("sdf", "temperature", "velx", "vely")
    AUGMENTATIONS = ("noise", "cell_centered")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation = self.model_cfg["params"].pop("augmentation", "noise")
        if self.augmentation not in self.AUGMENTATIONS:
            raise ValueError(f"augmentation must be one of {self.AUGMENTATIONS}, got {self.augmentation!r}")
        self.noise = LogUniformNoise(0.001, 5e-2, skip_prob=0.1)
        self.field_dropout = FieldDropout()  # p=0.1, active_prob=0.4, as in ModuleBase
        # Kept for code that inspects .augmentations; _augment does the work.
        self.augmentations = [self.noise] if self.augmentation == "noise" else [self.noise, self.field_dropout]

    @staticmethod
    def _model_input(data: DivFreeData) -> StaggeredFields:
        return StaggeredFields(sdf=data.sdf, temperature=data.temperature, velx=data.velx, vely=data.vely)

    @staticmethod
    def _data_cell_tensor(data: DivFreeData) -> torch.Tensor:
        return staggered_to_cells(data.sdf, data.temperature, data.velx, data.vely)

    def _augment_field(self, field: torch.Tensor) -> torch.Tensor:
        return self.noise(field.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def _augment_cell_centered(self, data: DivFreeData) -> DivFreeData:
        fields = [getattr(data, name) for name in self.INPUT_FIELDS]
        batch_size, device = fields[0].shape[0], fields[0].device
        per_sample = (batch_size, 1, 1, 1)

        log_min, log_max = math.log(self.noise.min), math.log(self.noise.max)
        scale = (torch.rand(batch_size, device=device) * (log_max - log_min) + log_min).exp()
        apply = torch.rand(batch_size, device=device) >= self.noise.skip_prob
        amplitude = (scale * apply).view(per_sample)
        fields = [field + torch.randn_like(field) * amplitude for field in fields]

        n_fields = len(fields)
        active = torch.rand(batch_size, n_fields, device=device) < self.field_dropout.active_prob
        keep = (torch.rand(batch_size, n_fields, device=device) > self.field_dropout.p).to(fields[0].dtype)
        mask = torch.where(active, keep, torch.ones_like(keep))
        fields = [field * mask[:, i].view(per_sample) for i, field in enumerate(fields)]
        return replace(data, **dict(zip(self.INPUT_FIELDS, fields)))

    def _augment(self, data: DivFreeData) -> DivFreeData:
        if self.augmentation == "cell_centered":
            return self._augment_cell_centered(data)
        return data._apply(self._augment_field)

    def _field_loss(self, output: Nucleus2MoEStaggeredOutput, target: DivFreeData) -> torch.Tensor:
        """MAE over all elements of the loss fields together (fields with more
        elements, the face velocities, weigh slightly more)."""
        total_abs_error = sum(
            (getattr(output, name) - getattr(target, name)).abs().sum() for name in self.LOSS_FIELDS
        )
        total_elements = sum(getattr(output, name).numel() for name in self.LOSS_FIELDS)
        return total_abs_error / total_elements

    def _per_field_mae(self, output: Nucleus2MoEStaggeredOutput, target: DivFreeData, prefix: str) -> dict:
        metrics = {
            f"{prefix}/mae_{name}": (getattr(output, name) - getattr(target, name)).abs().mean()
            for name in self.LOSS_FIELDS
        }
        # Diagnostic only: discrete divergence of predicted vs. true faces, in units of
        # the normalized velocity per grid spacing. Not part of the loss.
        div_pred = divergence_centers_from_faces(output.velx, output.vely, GRADIENT_SPACING, GRADIENT_SPACING)
        div_true = divergence_centers_from_faces(target.velx, target.vely, GRADIENT_SPACING, GRADIENT_SPACING)
        metrics[f"{prefix}/mae_divergence"] = (div_pred - div_true).abs().mean()
        return metrics

    def training_step(self, batch: DivFreeBatch, batch_idx: int) -> torch.Tensor:
        torch.compiler.cudagraph_mark_step_begin()
        model_input = self._model_input(self._augment(batch.input))
        output = self.model.step(model_input, batch.sim_params_tensor)

        data_loss = self._field_loss(output, batch.target)
        aux_loss, _ = self._router_loss(output.moe_outputs)
        loss = data_loss + aux_loss
        self._update_router_bias(output.moe_outputs)

        log_dict = {
            "train/loss": loss,
            "train/data_loss": data_loss,
            "train/step": self.global_step,
            "train/learning_rate": self.get_current_lr(),
        }
        log_dict |= self._per_field_mae(output, batch.target, "train")
        log_dict = self._moe_metrics(output.moe_outputs, log_dict, "train")
        self.default_log_dict(log_dict)
        return loss

    def validation_step(self, batch: DivFreeBatch, batch_idx: int) -> torch.Tensor:
        output = self.model.step(self._model_input(batch.input), batch.sim_params_tensor)
        loss = self._field_loss(output, batch.target)

        pred_cells = output.to_cell_tensor()
        target_cells = self._data_cell_tensor(batch.target)
        if batch_idx == 0:
            input_cells = self._data_cell_tensor(batch.input)
            self.validation_sample = (input_cells.detach(), target_cells.detach(), pred_cells.detach())

        log_dict = {"val/loss": loss, "val/data_loss": loss}
        log_dict |= self._per_field_mae(output, batch.target, "val")
        log_dict = self.log_step_metrics(
            log_dict, pred_cells, target_cells, batch.dx[0].item(), batch.dy[0].item(), "val"
        )
        log_dict = self._moe_metrics(output.moe_outputs, log_dict, "val")
        self.default_log_dict(log_dict)
        return loss
