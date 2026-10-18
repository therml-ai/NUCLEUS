from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.profiler import record_function
from typing import Optional, Tuple
from boiling_data.flashx.batch import FlashXBatch, unstack_field_cells
from boiling_data.flashx.normalizer import NormalizerWrapper

from nucleus.models.flashx_io import require_normalized
from nucleus.models.nucleus2_moe_base import Nucleus2MoEBase, Nucleus2MoEBaseConfig, config_from_dict, patch_debed
from nucleus.physics.coordinates import CellCenteredGrid, cell_centered_grid
from nucleus.physics.div_projection import div_projection
from nucleus.physics.velocity_boundaries import (
    FaceVelocityBoundary,
    apply_velocity_boundaries,
    flashx_face_velocity_boundary,
)
from nucleus.physics.sdf_reinit import sdf_reinit_sussman
from nucleus.physics.sdf import band_mask
from nucleus.physics.mass_transfer import continuity
from nucleus.physics.ansatz import temperature_ansatz

from ._api import register_model

__all__ = [
    "Nucleus2MoEDivFreeProjection",
    "Nucleus2MoEDivFreeProjectionConfig",
    "Nucleus2MoEDivFreeProjectionOutput",
    "DivFreePhysics",
]

FIELD_NAMES = ("dfun", "temperature", "velfacex", "velfacey")
FACE_FIELDS = ("velfacex", "velfacey")
# sdf, temperature, and each face velocity split onto the low and high face of its
# cells, as FlashXBatch.stack_field_cells lays them out
NUM_INPUT_CHANNELS = 6
# the input's channels, the face velocities' solenoidal part, and the divergence source
NUM_OUTPUT_CHANNELS = NUM_INPUT_CHANNELS + 1



@dataclass
class Nucleus2MoEDivFreeProjectionConfig(Nucleus2MoEBaseConfig):
    # the temperature is the network's, held at saturation on the interface and at
    # the heater temperature on the heater; otherwise the network predicts it alone.
    # It sets what the network learns, so unlike the options below it is fixed at init.
    use_temperature_ansatz: bool = False
    # the physics forward applies unless a call overrides it
    use_sdf_reinit: bool = False
    # the prediction's face velocity is the network's projected so its divergence
    # equals the (gated) divergence source, not the network's own
    use_div_projection: bool = False
    # zero the network's divergence source away from the interface before the
    # projection; the continuity source is zero there already, so use_mass_transfer
    # skips the gate
    use_div_gate: bool = False
    # take the divergence source from the continuity equation, not the network
    use_mass_transfer: bool = False
    # set the network's normal face velocity on each boundary from its boundary type:
    # zero on walls, divergence-free boundary cells on an outflow
    use_velocity_boundaries: bool = False


@dataclass
class Nucleus2MoEDivFreeProjectionOutput:
    """Prediction of :class:`Nucleus2MoEDivFreeProjection`. ``prediction`` holds dfun,
    temperature, velfacex and velfacey on their natural grids, its velocities
    velx_div_proj and vely_div_proj with use_div_projection and nn_velx_df and
    nn_vely_df otherwise. velx_div_proj and vely_div_proj are the network's face
    velocities projected so their divergence is gated_div_source. Everything is
    normalized, the velocities and divergences by the shared velocity scale.
    ``moe_outputs`` is the per-block MoE routing output for the router losses."""
    prediction: FlashXBatch
    nn_velx_df: torch.Tensor  # the network's x-face velocity
    nn_vely_df: torch.Tensor
    velx_div_proj: torch.Tensor  # nn_velx_df projected onto the gated source's divergence
    vely_div_proj: torch.Tensor
    div_source: torch.Tensor  # cell-centered divergence source, the network's or the continuity equation's
    gated_div_source: torch.Tensor  # div_source, but with zero gating applied away from interfaces
    moe_outputs: list
    grid: CellCenteredGrid


@dataclass(frozen=True)
class DivFreePhysics:
    """The per-sample parameters the physics reads, each ``(batch, 1, 1, 1)``, with
    temperatures non-dimensional like the BubbleML temperature field: 0 at the bulk
    temperature and 1 at the wall temperature scale."""
    sat_temp: torch.Tensor
    heater_temp: torch.Tensor
    heater_x_min: torch.Tensor
    heater_x_max: torch.Tensor
    contact_angle: torch.Tensor
    stefan: torch.Tensor
    reynolds: torch.Tensor
    prandtl: torch.Tensor
    thermal_conductivity: torch.Tensor
    rhogas: torch.Tensor
    grid: CellCenteredGrid
    # each sample's (left, right, bottom, top) face velocity conditions, when asked for
    velocity_boundaries: Tuple[Tuple[FaceVelocityBoundary, ...], ...] = ()


def heater_mask(physics: DivFreePhysics) -> torch.Tensor:
    """``(batch, 1, 1, width)``: the cells whose center lies strictly within the
    heater's x extent, where Flash-X applies the heater's wall temperature and
    contact angle."""
    x_centers = physics.grid.x_centers
    return (x_centers > physics.heater_x_min) & (x_centers < physics.heater_x_max)


def continuity_source(temp: torch.Tensor, sdf: torch.Tensor, physics: DivFreePhysics) -> torch.Tensor:
    """``div(u)`` from phase change at the interface, ``(..., H, W)``, from the
    temperature and sdf in the BubbleML files' units, computed as Flash-X does."""
    as_float64 = lambda tensor: tensor.to(torch.float64)
    return continuity(
        temp, sdf,
        sat_temp=as_float64(physics.sat_temp),
        dx=physics.grid.dx, dy=physics.grid.dy,
        stefan=as_float64(physics.stefan),
        reynolds=as_float64(physics.reynolds),
        prandtl=as_float64(physics.prandtl),
        thermal_conductivity=as_float64(physics.thermal_conductivity),
        rhogas=as_float64(physics.rhogas),
        wall_temp=as_float64(physics.heater_temp),
        contact_angle=as_float64(physics.contact_angle),
        heater_mask=heater_mask(physics),
    )


def divfree_physics(raw: FlashXBatch, velocity_boundaries: bool = False) -> DivFreePhysics:
    """The physics parameters of a raw (unnormalized) batch holding dfun, whose grid
    sets the cells, with the samples' face velocity conditions when
    velocity_boundaries is set."""
    parameter = lambda group, name: raw.parameter(group, name)[:, None, None, None]
    bulk_temp = parameter("physical", "bulk_temp")
    temperature_scale = parameter("physical", "wall_temp_scale") - bulk_temp
    return DivFreePhysics(
        sat_temp=(parameter("physical", "sat_temp") - bulk_temp) / temperature_scale,
        heater_temp=parameter("heaters", "wall_temp_fraction"),
        heater_x_min=parameter("heaters", "xMin"),
        heater_x_max=parameter("heaters", "xMax"),
        contact_angle=parameter("heaters", "advAngle"),
        stefan=parameter("non_dimensional", "stefan"),
        reynolds=1.0 / parameter("non_dimensional", "inv_reynolds"),
        prandtl=parameter("non_dimensional", "prandtl"),
        thermal_conductivity=parameter("non_dimensional", "thcogas"),
        rhogas=parameter("non_dimensional", "rhogas"),
        grid=cell_centered_grid(
            raw.parameter("discretization", "x_min"),
            shared_spacing(raw.dx("dfun"), "dx"),
            shared_spacing(raw.dy("dfun"), "dy"),
            raw.fields["dfun"].shape[-1],
        ),
        velocity_boundaries=(
            tuple(sample_velocity_boundaries(config) for config in raw.configs) if velocity_boundaries else ()
        ),
    )


def shared_spacing(spacing_per_sample: torch.Tensor, name: str) -> float:
    """The one spacing every sample of a batch has, from its ``(batch,)`` spacings."""
    spacing = float(spacing_per_sample[0])
    # the Poisson solves and finite differences take one spacing per call
    if not torch.allclose(spacing_per_sample, torch.full_like(spacing_per_sample, spacing)):
        raise ValueError(f"every sample in a batch must share one grid spacing, not {name} = {spacing_per_sample.tolist()}")
    return spacing


def sample_velocity_boundaries(config) -> Tuple[FaceVelocityBoundary, ...]:
    boundary = config.get("boundary", {})
    return tuple(
        flashx_face_velocity_boundary(boundary.get(side, {}).get("type"))
        for side in ("left", "right", "bottom", "top")
    )


def with_velocity_boundaries(velfacex, velfacey, physics: "DivFreePhysics"):
    """The face velocities with each sample's boundary conditions applied; a batch
    can mix conditions, e.g. a flipped flow boiling sample's inflow on the other side."""
    grid, conditions = physics.grid, physics.velocity_boundaries
    if len(set(conditions)) == 1:
        return apply_velocity_boundaries(velfacex, velfacey, grid.dx, grid.dy, *conditions[0])
    samples = [
        apply_velocity_boundaries(velfacex[index:index + 1], velfacey[index:index + 1], grid.dx, grid.dy, *condition)
        for index, condition in enumerate(conditions)
    ]
    return torch.cat([x for x, _ in samples]), torch.cat([y for _, y in samples])


def _config_from_dict(config_dict: dict) -> Nucleus2MoEDivFreeProjectionConfig:
    # checkpoints saved before the option existed were trained with the ansatz
    return config_from_dict(Nucleus2MoEDivFreeProjectionConfig, {"use_temperature_ansatz": True, **config_dict})


class ChannelsLastConv2d(nn.Module):
    """Conv2d over the trailing spatial dims of a channels-last (B, T, H, W, C)
    tensor. Time is folded into the batch so each frame is convolved
    independently (no temporal mixing), then the layout is restored.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, **conv_kwargs):
        super().__init__()
        conv_kwargs.setdefault("padding", kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, height, width, channels = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(batch * time, channels, height, width)
        x = self.conv(x)
        _, out_channels, out_height, out_width = x.shape
        x = x.reshape(batch, time, out_channels, out_height, out_width)
        return x.permute(0, 1, 3, 4, 2).contiguous()

@register_model("nucleus2_moe_divfree_projection")
class Nucleus2MoEDivFreeProjection(Nucleus2MoEBase):
    config_class = Nucleus2MoEDivFreeProjectionConfig
    config_from_dict = staticmethod(_config_from_dict)
    field_names = FIELD_NAMES
    # one shared scale and no offset, so the divergence, the projection and the
    # Poisson solve, all linear, commute with normalization
    vector_fields = (("velfacex", "velfacey"),)
    runtime_options = (
        "use_sdf_reinit", "use_div_projection", "use_div_gate", "use_mass_transfer", "use_velocity_boundaries",
    )

    def __init__(self, config: Nucleus2MoEDivFreeProjectionConfig, normalizer: NormalizerWrapper):
        super().__init__(config, normalizer, in_channels=NUM_INPUT_CHANNELS)
        self.debed = patch_debed(config, out_channels=NUM_OUTPUT_CHANNELS)

    def forward(
        self,
        batch: FlashXBatch,
        use_sdf_reinit: Optional[bool] = None,
        use_div_projection: Optional[bool] = None,
        use_div_gate: Optional[bool] = None,
        use_mass_transfer: Optional[bool] = None,
        use_velocity_boundaries: Optional[bool] = None,
    ) -> Nucleus2MoEDivFreeProjectionOutput:
        """batch: normalized, holding field_names and a config tensor of config_keys.
        A physics option left as None takes its value from the model's config."""
        require_normalized(batch)
        use_sdf_reinit = self._option(use_sdf_reinit, "use_sdf_reinit")
        use_div_projection = self._option(use_div_projection, "use_div_projection")
        use_div_gate = self._option(use_div_gate, "use_div_gate")
        use_mass_transfer = self._option(use_mass_transfer, "use_mass_transfer")
        use_velocity_boundaries = self._option(use_velocity_boundaries, "use_velocity_boundaries")
        # the spacing needs the raw configs and the grid of one field, so only dfun is unnormalized
        raw = self.normalizer.unnormalize(batch.with_fields({"dfun": batch.fields["dfun"]}))
        physics: DivFreePhysics = divfree_physics(raw, use_velocity_boundaries)
        x, moe_outputs = self.model_step(batch.stack_field_cells(self.field_names), batch.config_tensor())
        nn_sdf, nn_temp, nn_vel_top, nn_vel_bot, nn_vel_left, nn_vel_right, nn_div_source = x.unbind(-1)
        faces = unstack_field_cells(
            torch.stack((nn_vel_left, nn_vel_right, nn_vel_bot, nn_vel_top), dim=-1), FACE_FIELDS
        )
        nn_velx_df, nn_vely_df = faces["velfacex"], faces["velfacey"]
        if use_velocity_boundaries:
            # before the projection, which keeps the wall faces as they are; the
            # velocities are normalized by a pure scale, so the linear conditions
            # apply to normalized values directly
            # NOTE: this is only safe if the velocities normalization does not shift
            nn_velx_df, nn_vely_df = with_velocity_boundaries(nn_velx_df, nn_vely_df, physics)
        grid = physics.grid

        nn_sdf_physical = self.normalizer.unnormalize_field("dfun", nn_sdf)
        if use_sdf_reinit:
            sdf_physical = sdf_reinit_sussman(
                nn_sdf_physical, grid.dx, grid.dy, n_iter=5,
                contact_angle=physics.contact_angle, heater_mask=heater_mask(physics),
            )
        else:
            sdf_physical = nn_sdf_physical
        sdf = self.normalizer.normalize_field("dfun", sdf_physical)

        # construct temp, possibly using asnatz solution
        temp = self._temperature(nn_temp, nn_sdf_physical, physics)

        # Find the RHS of the continuity equation.
        # if use_mass_transfer, this is computed as (mass_transfer * dot(n, grad(rho)))
        # otherwise, use the neural networks output `div_source`
        if use_mass_transfer:
            temp_physical = self.normalizer.unnormalize_field("temperature", temp)
            div_source = self._continuity_div_source(
                # the divergence source supervises the velocity, not the interface: its
                # gradients through the level set's normals and densities are stiff
                temp_physical.to(torch.float64), sdf_physical.detach().to(torch.float64), physics
            )
        else:
            div_source = nn_div_source

        # if `use_mass_transfer`, the continuity source is already zero away from the interface, so only the
        # network's source is gated
        if use_div_gate and not use_mass_transfer:
            div_gate = band_mask(sdf_physical, 3.0 * grid.dx).to(self.debed_dtype)
            gated_div_source = div_gate * div_source
        else:
            gated_div_source = div_source

        # The velocities are normalized by a pure scale and the projection is linear,
        # so it acts on normalized values directly: the velocity whose divergence is
        # the source, nearest the network's.
        velx_div_proj, vely_div_proj = div_projection(
            nn_velx_df, nn_vely_df, grid.dx, grid.dy, source=gated_div_source
        )

        velfacex, velfacey = (velx_div_proj, vely_div_proj) if use_div_projection else (nn_velx_df, nn_vely_df)
        prediction = batch.with_fields({
            "dfun": sdf.to(torch.float32),
            "temperature": temp.to(torch.float32),
            "velfacex": velfacex.to(torch.float32),
            "velfacey": velfacey.to(torch.float32),
        })
        return Nucleus2MoEDivFreeProjectionOutput(
            prediction=prediction,
            nn_velx_df=nn_velx_df.to(torch.float32),
            nn_vely_df=nn_vely_df.to(torch.float32),
            velx_div_proj=velx_div_proj.to(torch.float32),
            vely_div_proj=vely_div_proj.to(torch.float32),
            div_source=div_source.to(torch.float32),
            gated_div_source=gated_div_source.to(torch.float32),
            moe_outputs=moe_outputs,
            grid=grid,
        )

    def _option(self, value: Optional[bool], name: str) -> bool:
        return getattr(self.config, name) if value is None else value

    def velocity_scale(self) -> float:
        offset_x, scale = self.normalizer.field_offset_and_scale("velfacex")
        offset_y, scale_y = self.normalizer.field_offset_and_scale("velfacey")
        if (offset_x, offset_y) != (0.0, 0.0) or scale != scale_y:
            raise ValueError("velfacex and velfacey must be normalized as one of the normalizer's vector_fields")
        return scale

    def _temperature(self, nn_temp, nn_sdf_physical, physics: DivFreePhysics) -> torch.Tensor:
        """The normalized temperature, the network's own without use_temperature_ansatz."""
        if not self.config.use_temperature_ansatz:
            return nn_temp
        grid = physics.grid
        # temperature_ansatz adds `nn_temp` directly to sat_temp/heater_temp (field =
        # sat_temp + decay * nn), so those two are normalized like nn_temp rather than
        # unnormalizing them all.
        # The network's own sdf, not the reinitialized one: nn_temp was learned against
        # it, and reinitializing rescales the band where the decay multiplies nn_temp.
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

    def _continuity_div_source(self, temp_physical, sdf_physical, physics: DivFreePhysics) -> torch.Tensor:
        """The normalized divergence source from phase change at the interface, from the
        temperature and sdf in the BubbleML files' units."""
        return continuity_source(temp_physical, sdf_physical, physics) / self.velocity_scale()

    def model_step(self, cells: torch.Tensor, sim_params: torch.Tensor):
        """cells: ``(B, T, H, W, 6)`` as FlashXBatch.stack_field_cells lays out
        field_names. Returns the ``(B, T, H, W, 7)`` network output, which is sdf,
        temperature, each cell's top, bottom, left and right face velocities, and the
        divergence source, then the MoE outputs."""
        _, _, h, w, _ = cells.shape
        x, moe_outputs = self.process(cells, sim_params)
        with record_function("debed"):
            x = self.debed(x, target_shape=(h, w))
        return x, moe_outputs
