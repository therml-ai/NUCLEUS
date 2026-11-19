from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper

from nucleus.layers.moe.topk_moe import TopkMoEOutput
from nucleus.models.flashx_io import num_cell_channels, require_normalized
from nucleus.models.nucleus2_moe_base import Nucleus2MoEBase, Nucleus2MoEBaseConfig, config_from_dict, patch_debed
from nucleus.models.nucleus2_moe_divfree_projection import (
    DivFreePhysics,
    continuity_source,
    divfree_physics,
    heater_mask,
)
from nucleus.physics.ansatz import temperature_ansatz
from nucleus.physics.coordinates import CellCenteredGrid
from nucleus.physics.poisson import curl_faces_from_nodes, grad_faces_from_centers, solve_poisson_neumann_dirichlet
from nucleus.physics.sdf import band_mask
from nucleus.physics.sdf_reinit import sdf_reinit_sussman

from ._api import register_model

__all__ = [
    "Nucleus2MoEDivFreeCurl",
    "Nucleus2MoEDivFreeCurlConfig",
    "Nucleus2MoEDivFreeCurlOutput",
    "nodes_from_cell_corners",
]

FIELD_NAMES = ("dfun", "temperature", "velfacex", "velfacey")
NUM_INPUT_CHANNELS = num_cell_channels(FIELD_NAMES)
# each cell's (row, column) offset to its bottom-left, bottom-right, top-left and
# top-right nodes, the order of the stream function's channels
CORNER_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
# sdf, temperature, the stream function at each cell's corners, and the divergence source
NUM_OUTPUT_CHANNELS = 2 + len(CORNER_OFFSETS) + 1


@dataclass
class Nucleus2MoEDivFreeCurlConfig(Nucleus2MoEBaseConfig):
    # the temperature is the network's, held at saturation on the interface and at
    # the heater temperature on the heater; otherwise the network predicts it alone.
    # It sets what the network learns, so unlike the options below it is fixed at init.
    use_temperature_ansatz: bool = False
    # the physics forward applies unless a call overrides it
    use_sdf_reinit: bool = False
    # zero the network's divergence source away from the interface; the continuity
    # source is zero there already, so use_mass_transfer skips the gate
    use_div_gate: bool = False
    # take the divergence source from the continuity equation, not the network
    use_mass_transfer: bool = False


@dataclass
class Nucleus2MoEDivFreeCurlOutput:
    prediction: FlashXBatch
    # (B, T, H + 1, W + 1) on the nodes, in normalized velocity times the cell width dx
    psi_nodes: torch.Tensor
    solenoidal_velx_faces: torch.Tensor  # (B, T, H, W + 1), the curl of the stream function
    solenoidal_vely_faces: torch.Tensor  # (B, T, H + 1, W)
    dilatational_velx_faces: torch.Tensor  # (B, T, H, W + 1), the gradient of the potential whose laplacian is gated_div_source
    dilatational_vely_faces: torch.Tensor  # (B, T, H + 1, W)
    div_source: torch.Tensor  # (B, T, H, W), the network's or the continuity equation's
    gated_div_source: torch.Tensor  # (B, T, H, W), div_source zeroed away from interfaces with use_div_gate
    moe_outputs: List[TopkMoEOutput]
    grid: CellCenteredGrid


def nodes_from_cell_corners(corners: torch.Tensor) -> torch.Tensor:
    """``(..., H + 1, W + 1)`` node values from ``(..., H, W, 4)`` per-cell corner
    values laid out as CORNER_OFFSETS. A node is a corner of up to four cells, so
    their predictions of it are averaged, as unstack_field_cells averages faces."""
    total = count = 0
    for channel, (row, column) in enumerate(CORNER_OFFSETS):
        padding = (column, 1 - column, row, 1 - row)
        total = total + F.pad(corners[..., channel], padding)
        count = count + F.pad(torch.ones_like(corners[..., channel]), padding)
    return total / count


def solenoidal_velocity(psi_nodes: torch.Tensor, grid: CellCenteredGrid) -> Tuple[torch.Tensor, torch.Tensor]:
    # the stream function is in normalized velocity times a cell width
    return curl_faces_from_nodes(psi_nodes.to(torch.float64) * grid.dx, grid.dx, grid.dy)


def dilatational_velocity(div_source: torch.Tensor, grid: CellCenteredGrid) -> Tuple[torch.Tensor, torch.Tensor]:
    """The float64 face velocity grad(phi), where laplacian(phi) = div_source with
    Neumann walls and a Dirichlet top outflow, so its divergence is div_source."""
    # float64 keeps the DCT-based solve's residual well below float32 noise
    phi_centers = solve_poisson_neumann_dirichlet(div_source.to(torch.float64), grid.dx, grid.dy)
    return grad_faces_from_centers(phi_centers, grid.dx, grid.dy)


def _config_from_dict(config_dict: dict) -> Nucleus2MoEDivFreeCurlConfig:
    return config_from_dict(Nucleus2MoEDivFreeCurlConfig, config_dict)


@register_model("nucleus2_moe_divfree_curl")
class Nucleus2MoEDivFreeCurl(Nucleus2MoEBase):
    config_class = Nucleus2MoEDivFreeCurlConfig
    config_from_dict = staticmethod(_config_from_dict)
    field_names = FIELD_NAMES
    # one shared scale and no offset, so the curl, the divergence and the Poisson
    # solve, all linear, commute with normalization
    vector_fields = (("velfacex", "velfacey"),)
    runtime_options = ("use_sdf_reinit", "use_div_gate", "use_mass_transfer")

    def __init__(self, config: Nucleus2MoEDivFreeCurlConfig, normalizer: NormalizerWrapper):
        super().__init__(config, normalizer, in_channels=NUM_INPUT_CHANNELS)
        self.debed = patch_debed(config, out_channels=NUM_OUTPUT_CHANNELS)

    def forward(
        self,
        batch: FlashXBatch,
        use_sdf_reinit: Optional[bool] = None,
        use_div_gate: Optional[bool] = None,
        use_mass_transfer: Optional[bool] = None,
    ) -> Nucleus2MoEDivFreeCurlOutput:
        """batch: normalized, holding field_names and a config tensor of config_keys.
        A physics option left as `None` will use the value set in the model's config."""
        require_normalized(batch)
        # the curl and the Poisson solve act on normalized velocities, which is only
        # exact when both components share one scale and no offset
        velocity_scale = self.velocity_scale()
        use_sdf_reinit = self._option(use_sdf_reinit, "use_sdf_reinit")
        use_div_gate = self._option(use_div_gate, "use_div_gate")
        use_mass_transfer = self._option(use_mass_transfer, "use_mass_transfer")
        # the spacing needs the raw configs and the grid of one field, so only dfun is unnormalized
        physics = divfree_physics(self.normalizer.unnormalize(batch.with_fields({"dfun": batch.fields["dfun"]})))
        grid = physics.grid
        network, moe_outputs = self.step(batch.stack_field_cells(self.field_names), batch.config_tensor())
        nn_sdf, nn_temp = network[..., 0], network[..., 1]
        psi_corners = network[..., 2:2 + len(CORNER_OFFSETS)]
        psi_nodes = nodes_from_cell_corners(psi_corners)
        nn_div_source = network[..., -1]

        nn_sdf_physical = self.normalizer.unnormalize_field("dfun", nn_sdf)
        if use_sdf_reinit:
            sdf_physical = sdf_reinit_sussman(
                nn_sdf_physical, grid.dx, grid.dy, n_iter=5,
                contact_angle=physics.contact_angle, heater_mask=heater_mask(physics),
            )
        else:
            sdf_physical = nn_sdf_physical
        temp = self._temperature(nn_temp, nn_sdf_physical, physics)

        if use_mass_transfer:
            temp_physical = self.normalizer.unnormalize_field("temperature", temp)
            div_source = continuity_source(
                temp_physical.to(torch.float64), sdf_physical.detach().to(torch.float64), physics
            ) / velocity_scale
        else:
            div_source = nn_div_source

        if use_div_gate and not use_mass_transfer:
            gated_div_source = band_mask(sdf_physical, 3.0 * grid.dx).to(div_source.dtype) * div_source
        else:
            gated_div_source = div_source

        solenoidal_velx_faces, solenoidal_vely_faces = solenoidal_velocity(psi_nodes, grid)
        dilatational_velx_faces, dilatational_vely_faces = dilatational_velocity(gated_div_source, grid)
        prediction = batch.with_fields({
            "dfun": self.normalizer.normalize_field("dfun", sdf_physical).to(torch.float32),
            "temperature": temp.to(torch.float32),
            "velfacex": (solenoidal_velx_faces + dilatational_velx_faces).to(torch.float32),
            "velfacey": (solenoidal_vely_faces + dilatational_vely_faces).to(torch.float32),
        })
        return Nucleus2MoEDivFreeCurlOutput(
            prediction=prediction,
            psi_nodes=psi_nodes.to(torch.float32),
            solenoidal_velx_faces=solenoidal_velx_faces.to(torch.float32),
            solenoidal_vely_faces=solenoidal_vely_faces.to(torch.float32),
            dilatational_velx_faces=dilatational_velx_faces.to(torch.float32),
            dilatational_vely_faces=dilatational_vely_faces.to(torch.float32),
            div_source=div_source.to(torch.float32),
            gated_div_source=gated_div_source.to(torch.float32),
            moe_outputs=moe_outputs,
            grid=grid,
        )

    def _option(self, value: Optional[bool], name: str) -> bool:
        return getattr(self.config, name) if value is None else value

    def velocity_scale(self) -> float:
        offset_x, scale_x = self.normalizer.field_offset_and_scale("velfacex")
        offset_y, scale_y = self.normalizer.field_offset_and_scale("velfacey")
        if (offset_x, offset_y) != (0.0, 0.0) or scale_x != scale_y:
            raise ValueError("velfacex and velfacey must be normalized as one of the normalizer's vector_fields")
        return scale_x

    def _temperature(self, nn_temp: torch.Tensor, nn_sdf_physical: torch.Tensor, physics: DivFreePhysics) -> torch.Tensor:
        """The normalized temperature, the network's own without use_temperature_ansatz."""
        if not self.config.use_temperature_ansatz:
            return nn_temp
        grid = physics.grid
        # sat_temp and heater_temp are normalized like nn_temp, which the ansatz adds to them
        return temperature_ansatz(
            nn_temp,
            nn_sdf_physical.detach(),
            self.normalizer.normalize_field("temperature", physics.sat_temp),
            band_width=2 * grid.dx,
            heater_temperature=self.normalizer.normalize_field("temperature", physics.heater_temp),
            x_coords=grid.x_centers,
            heater_x_min=physics.heater_x_min,
            heater_x_max=physics.heater_x_max,
            heater_band_width=2 * grid.dx,
            dy=grid.dy,
        )
