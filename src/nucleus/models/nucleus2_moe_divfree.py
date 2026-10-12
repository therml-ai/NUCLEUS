import dataclasses
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.profiler import record_function
from rotary_embedding_torch import RotaryEmbedding
from typing import List, Literal
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper

from nucleus.layers.adaptive_layernorm import AdaptiveLayerNorm
from nucleus.layers.attention import NeighborhoodAttention
from nucleus.layers.moe.topk_moe import TopkMoE, TopkMoEOutput, TopkRouterWithBias
from nucleus.layers.droppath import DropPath
from nucleus.layers import (
    AdaptiveEmbed,
    AdaptiveDebed,
    LinearEmbed,
    LinearDebed,
)
from nucleus.models.flashx_io import (
    NUCLEUS2_CONFIG_KEYS,
    num_config_columns,
    require_normalized,
    unnormalized_prediction,
)
from nucleus.physics.coordinates import CellGrid, cell_grid
from nucleus.physics.div_projection import div_projection
from nucleus.utils.sdf_reinit import sdf_reinit_sussman
from nucleus.physics.poisson import solve_poisson_neumann_dirichlet, grad_faces_from_centers
from nucleus.physics.sdf import band_mask
from nucleus.physics.mass_transfer import continuity
from nucleus.physics.ansatz import temperature_ansatz

from ._api import register_model

__all__ = [
    "Nucleus2MoEDivFree",
    "Nucleus2MoEDivFreeConfig",
    "Nucleus2MoEDivFreeOutput",
    "DivFreePhysics",
]

FIELD_NAMES = ("dfun", "temperature", "velfacex", "velfacey")
# sdf, temperature, and each face velocity split onto the low and high face of its
# cells, as FlashXBatch.stack_field_cells lays them out
NUM_INPUT_CHANNELS = 6


_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
_STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}


NUCLEUS_DTYPE = Literal["float32", "float16", "bfloat16"]


@dataclass
class Nucleus2MoEDivFreeConfig:
    patch_size: int
    embed_dim: int
    num_heads: int
    processor_blocks: int
    num_experts: int
    topk: int
    moe_intermediate_dim: int
    patching: Literal["Linear", "Adaptive"]
    embed_dtype: NUCLEUS_DTYPE = "float32"
    debed_dtype: NUCLEUS_DTYPE = "float32"
    activation_dtype: NUCLEUS_DTYPE = "float32"
    attention_dtype: NUCLEUS_DTYPE = "bfloat16"
    moe_dtype: NUCLEUS_DTYPE = "bfloat16"


@dataclass
class Nucleus2MoEDivFreeOutput:
    """Prediction of :class:`Nucleus2MoEDivFree`. ``prediction`` holds dfun,
    temperature, velfacex and velfacey on their natural grids; it and every tensor
    here are normalized, the velocities and their parts by the shared velocity
    scale, and phi so that its gradient is the normalized dilational velocity.
    ``moe_outputs`` is the per-block MoE routing output for the router losses."""
    prediction: FlashXBatch
    velx_sol: torch.Tensor
    vely_sol: torch.Tensor
    velx_dil: torch.Tensor
    vely_dil: torch.Tensor
    phi: torch.Tensor # grad(phi) is component of velocity contributed divergence
    div_source: torch.Tensor # cell-centered divergence source, RHS of poisson solve for phi
    gated_div_source: torch.Tensor # div_source, but with zero gating applied away from interfaces
    moe_outputs: list
    grid: CellGrid


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
    grid: CellGrid


def divfree_physics(raw: FlashXBatch, height: int, width: int) -> DivFreePhysics:
    """The physics parameters of a batch whose configs are raw (unnormalized)."""
    parameter = lambda group, name: raw.parameter(group, name)[:, None, None, None]
    bulk_temp = parameter("physical", "bulk_temp")
    temperature_scale = parameter("physical", "wall_temp_scale") - bulk_temp
    bounds = [raw.parameter("discretization", name) for name in ("x_min", "x_max", "y_min", "y_max")]
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
        grid=cell_grid(*bounds, height, width),
    )


def _config_to_dict(config: Nucleus2MoEDivFreeConfig) -> dict:
    d = dataclasses.asdict(config)
    dtype_fields = {"embed_dtype", "debed_dtype", "activation_dtype", "attention_dtype", "moe_dtype"}
    for key in dtype_fields:
        if not isinstance(d[key], str):
            d[key] = _DTYPE_TO_STR[d[key]]
    return d


def _config_from_dict(d: dict) -> Nucleus2MoEDivFreeConfig:
    d = dict(d)
    dtype_fields = {"embed_dtype", "debed_dtype", "activation_dtype", "attention_dtype", "moe_dtype"}
    for key in dtype_fields:
        if not isinstance(d[key], torch.dtype):
            d[key] = _STR_TO_DTYPE[d[key]]
    return Nucleus2MoEDivFreeConfig(**d)

def get_dtype(dtype):
    if isinstance(dtype, str):
        return _STR_TO_DTYPE[dtype]
    return dtype


class TransformerMoEBlock(nn.Module):
    def __init__(self, config: Nucleus2MoEDivFreeConfig, num_sim_params: int, drop_path_prob: float):
        super().__init__()

        self.activation_dtype = get_dtype(config.activation_dtype)
        self.attention_dtype = get_dtype(config.attention_dtype)
        self.moe_dtype = get_dtype(config.moe_dtype)

        self.drop_path = DropPath(drop_path_prob)

        self.attention_norm = AdaptiveLayerNorm(config.embed_dim, num_sim_params, dtype=self.attention_dtype)
        self.mlp_norm = AdaptiveLayerNorm(config.embed_dim, num_sim_params, self.moe_dtype)

        self.router = TopkRouterWithBias(
            config.num_experts,
            config.embed_dim,
            config.topk,
            bias_update_rate=0.001,
            softmax_first=False
        )

        self.attention = NeighborhoodAttention(embed_dim=config.embed_dim, num_heads=config.num_heads)

        self.mlp = TopkMoE(
            num_experts=config.num_experts,
            hidden_dim=config.embed_dim,
            intermediate_dim=config.moe_intermediate_dim,
            topk=config.topk,
            router=self.router
        )

    def _attention(self, x: torch.Tensor, freqs: torch.Tensor, sim_params: torch.Tensor) -> torch.Tensor:
        with record_function("attention"):
            h = x.to(self.attention_dtype)
            h = self.attention_norm(h, sim_params)
            h = self.attention(h, freqs)
            h = self.drop_path(h)
            x = x + h.to(self.activation_dtype)
        return x

    def _mlp(self, x: torch.Tensor, sim_params: torch.Tensor):
        with record_function("moe"):
            h = x.to(self.moe_dtype)
            h = self.mlp_norm(h, sim_params)
            moe_output: TopkMoEOutput = self.mlp(h)
            h = self.drop_path(moe_output.out)
            x = x + h.to(self.activation_dtype)
        return x, moe_output

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, sim_params: torch.Tensor):
        x = self._attention(x, freqs, sim_params)
        x, moe_output = self._mlp(x, sim_params)
        return x, moe_output

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

def cells_to_x_face(cells: torch.Tensor) -> torch.Tensor:
    """Interpolate a cell-centered field ``(..., H, W)`` to the x-faces ``(..., H, W+1)``:
    interior faces average their two bordering cells; the left/right wall faces take the
    single edge cell."""
    interior = 0.5 * (cells[..., :, :-1] + cells[..., :, 1:])
    return torch.cat([cells[..., :, :1], interior, cells[..., :, -1:]], dim=-1)


def cells_to_y_face(cells: torch.Tensor) -> torch.Tensor:
    """Interpolate a cell-centered field ``(..., H, W)`` to the y-faces ``(..., H+1, W)``."""
    interior = 0.5 * (cells[..., :-1, :] + cells[..., 1:, :])
    return torch.cat([cells[..., :1, :], interior, cells[..., -1:, :]], dim=-2)


@register_model("nucleus2_moe_divfree")
class Nucleus2MoEDivFree(nn.Module):
    config_class = Nucleus2MoEDivFreeConfig
    config_from_dict = staticmethod(_config_from_dict)
    config_keys = NUCLEUS2_CONFIG_KEYS
    field_names = FIELD_NAMES
    # one shared scale and no offset, so the divergence, the projection and the
    # Poisson solve, all linear, commute with normalization
    vector_fields = (("velfacex", "velfacey"),)
    num_sim_params = num_config_columns(NUCLEUS2_CONFIG_KEYS)

    def __init__(self, config: Nucleus2MoEDivFreeConfig, normalizer: NormalizerWrapper):
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        n_fields = NUM_INPUT_CHANNELS

        self.embed_dtype = get_dtype(config.embed_dtype)
        self.debed_dtype = get_dtype(config.debed_dtype)

        self.rotary_emb = RotaryEmbedding(
            dim=(config.embed_dim // config.num_heads) // 3,
            freqs_for="pixel",
            max_freq=256,
            seq_before_head_dim=True
        )

        drop_path_probs = torch.linspace(0.0, 0.1, config.processor_blocks)
        self.blocks = nn.ModuleList([
            TransformerMoEBlock(
                config=config,
                num_sim_params=self.num_sim_params,
                drop_path_prob=drop_path_probs[idx].item(),
            )
            for idx in range(config.processor_blocks)
        ])

        self.out_norm = nn.RMSNorm(config.embed_dim, dtype=self.debed_dtype)

        assert config.patching in ("Linear", "Adaptive")
        if config.patching == "Linear":
            self.embed = LinearEmbed(
                patch_size=config.patch_size,
                in_channels=n_fields,
                embed_dim=config.embed_dim,
                dtype=self.embed_dtype
            )
            self.debed = LinearDebed(
                patch_size=config.patch_size,
                embed_dim=config.embed_dim,
                out_channels=5,
                dtype=self.debed_dtype
            )
        else:
            self.embed = AdaptiveEmbed(
                in_channels=n_fields,
                out_channels=config.embed_dim,
                out_shape=(16, 16),
                dtype=config.embed_dtype
            )
            self.debed = AdaptiveDebed(
                in_channels=config.embed_dim,
                out_channels=5,
                patch_shape=(16, 16),
                dtype=config.debed_dtype
            )

    def get_extra_state(self):
        return {"model_name": getattr(self, "_model_name", None), "config": _config_to_dict(self.config)}

    def set_extra_state(self, state):
        self._model_name = state.get("model_name")
        self.config = _config_from_dict(state["config"])

    def forward(
        self,
        batch: FlashXBatch,
        use_sdf_reinit: bool = False,
        use_div_projection: bool = False,
        use_div_gate: bool = False,
        use_mass_transfer: bool = False,
    ) -> Nucleus2MoEDivFreeOutput:
        """batch: normalized, holding field_names and a config tensor of config_keys."""
        require_normalized(batch)
        height, width = batch.fields["dfun"].shape[-2:]
        physics = divfree_physics(self.normalizer.unnormalize(batch.with_fields({})), height, width)
        x, moe_outputs = self.model_step(batch.stack_field_cells(self.field_names), batch.config_tensor())
        nn_sdf, nn_temp, nn_velx_df, nn_vely_df, nn_div_source = x.unbind(-1)
        # The debed predicts velocity cell-centered (H, W); the reconstruction (div projection,
        # grad(phi), output) lives on the MAC faces, so interpolate to faces here.
        # NOTE: placeholder bridge -- revisit whether velocity should be predicted on
        # faces directly or the whole pipeline moved to cell centers.
        nn_velx_df = cells_to_x_face(nn_velx_df)
        nn_vely_df = cells_to_y_face(nn_vely_df)
        grid = physics.grid

        nn_sdf_physical = self.normalizer.unnormalize_field("dfun", nn_sdf)
        if use_sdf_reinit:
            sdf_physical = sdf_reinit_sussman(nn_sdf_physical, grid.dx, n_iter=5, near_threshold=0.1)
        else:
            sdf_physical = nn_sdf_physical
        sdf = self.normalizer.normalize_field("dfun", sdf_physical)

        # temperature_ansatz adds nn directly to sat_temp/heater_temp (field =
        # sat_temp + decay * nn), so those two are normalized like nn rather than
        # unnormalizing nn, so temp stays normalized throughout
        temp = temperature_ansatz(
            nn_temp,
            sdf_physical.detach(),
            self.normalizer.normalize_field("temperature", physics.sat_temp),
            band_width=2 * grid.dx,
            heater_temperature=self.normalizer.normalize_field("temperature", physics.heater_temp),
            x_coords=grid.x_centers,
            heater_x_min=physics.heater_x_min,
            heater_x_max=physics.heater_x_max,
            heater_band_width=2 * grid.dx,
            dy=grid.dy,
        )

        # The velocities are normalized by a pure scale and the projection, the
        # Poisson solve and the gradient are linear, so they act on normalized
        # values directly.
        if use_div_projection:
            velfacex_sol, velfacey_sol = div_projection(nn_velx_df, nn_vely_df, grid.dx, grid.dy)
        else:
            velfacex_sol = nn_velx_df
            velfacey_sol = nn_vely_df

        # Find the RHS of the continuity equation: (mass_transfer * dot(n, grad(rho)))
        if use_mass_transfer:
            div_source = self._continuity_div_source(temp.to(torch.float64), sdf_physical.to(torch.float64), physics)
        else:
            div_source = nn_div_source

        if use_div_gate:
            div_gate = band_mask(sdf_physical, 3.0 * grid.dx).to(self.debed_dtype)
            gated_div_source = div_gate * div_source
        else:
            gated_div_source = div_source

        phi = solve_poisson_neumann_dirichlet(gated_div_source.to(torch.float64), grid.dx, grid.dy)
        velfacex_dil, velfacey_dil = grad_faces_from_centers(phi, grid.dx, grid.dy)
        velfacex = velfacex_sol + velfacex_dil.to(torch.float32)
        velfacey = velfacey_sol + velfacey_dil.to(torch.float32)

        prediction = batch.with_fields({
            "dfun": sdf.to(torch.float32),
            "temperature": temp.to(torch.float32),
            "velfacex": velfacex.to(torch.float32),
            "velfacey": velfacey.to(torch.float32),
        })
        return Nucleus2MoEDivFreeOutput(
            prediction=prediction,
            velx_sol=velfacex_sol.to(torch.float32),
            vely_sol=velfacey_sol.to(torch.float32),
            velx_dil=velfacex_dil.to(torch.float32),
            vely_dil=velfacey_dil.to(torch.float32),
            phi=phi.to(torch.float32),
            div_source=div_source.to(torch.float32),
            gated_div_source=gated_div_source.to(torch.float32),
            moe_outputs=moe_outputs,
            grid=grid,
        )

    @torch.no_grad()
    def predict_next(self, window: FlashXBatch, num_timesteps: int) -> FlashXBatch:
        """The num_timesteps frames that follow a raw (unnormalized) window, raw. The
        rollout projects the solenoidal velocity and gates the divergence source,
        which training only encourages through the loss."""
        output = self(self.normalizer.normalize(window), use_div_projection=True, use_div_gate=True)
        return unnormalized_prediction(self.normalizer, window, output.prediction.tail_time_window(num_timesteps))

    def velocity_scale(self) -> float:
        offset_x, scale = self.normalizer.field_offset_and_scale("velfacex")
        offset_y, scale_y = self.normalizer.field_offset_and_scale("velfacey")
        if (offset_x, offset_y) != (0.0, 0.0) or scale != scale_y:
            raise ValueError("velfacex and velfacey must be normalized as one of the normalizer's vector_fields")
        return scale

    def _continuity_div_source(self, temp, sdf_physical, physics: DivFreePhysics) -> torch.Tensor:
        """The normalized divergence source from phase change at the interface."""
        temp_nd = self.normalizer.unnormalize_field("temperature", temp)
        as_float64 = lambda tensor: tensor.to(torch.float64)
        divergence = continuity(
            temp_nd, sdf_physical,
            sat_temp=as_float64(physics.sat_temp),
            dx=physics.grid.dx, dy=physics.grid.dy,
            stefan=as_float64(physics.stefan),
            reynolds=as_float64(physics.reynolds),
            prandtl=as_float64(physics.prandtl),
            thermal_conductivity=as_float64(physics.thermal_conductivity),
            rhogas=as_float64(physics.rhogas),
            wall_temp=as_float64(physics.heater_temp),
            contact_angle=as_float64(physics.contact_angle),
        )
        return divergence / self.velocity_scale()

    def model_step(self, cells: torch.Tensor, sim_params: torch.Tensor):
        """cells: ``(B, T, H, W, 6)`` as FlashXBatch.stack_field_cells lays out
        field_names. Returns the ``(B, T, H, W, 5)`` network output and the MoE outputs."""
        assert cells.dtype == torch.float32, f"expected float32, got {cells.dtype}"

        _, _, h, w, _ = cells.shape

        with record_function("encode"):
            x = embed = self.embed(cells.to(self.embed_dtype))

        x = x.to(get_dtype(self.config.activation_dtype))

        with record_function("get_axial_freqs"):
            with torch.no_grad():
                _, embed_t, embed_h, embed_w, _ = embed.shape
                rotary_freqs = self.rotary_emb.get_axial_freqs(embed_t, embed_h, embed_w)[None, :, :, :, None, :]

        moe_outputs = []
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
                x, moe_output = blk(x, rotary_freqs, sim_params)
                moe_outputs.append(moe_output)

        with record_function("debed"):
            x = self.out_norm(x.to(self.debed_dtype) + embed.to(self.debed_dtype))
            x = self.debed(x, target_shape=(h, w))

        return x, moe_outputs
