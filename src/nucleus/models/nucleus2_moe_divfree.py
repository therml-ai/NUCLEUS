import dataclasses
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.profiler import record_function
from rotary_embedding_torch import RotaryEmbedding
from typing import Dict, List, Literal, Optional
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger

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
from nucleus.data.in_mem_divfree_forecast_dataset import DivFreeBatch, DivFreeData
from nucleus.physics.leray import leray_projection
from nucleus.trajectory import Trajectory
from nucleus.models.modules import MoEConditionedForecastModule
from nucleus.noise import LogUniformNoise
from nucleus.utils.sdf_reinit import sdf_reinit_sussman
from nucleus.utils.inf_stabilizer import clip_temp_by_phase
from nucleus.physics.poisson import (
    solve_poisson_neumann_dirichlet,
    stream_function_from_faces,
    divergence_centers_from_faces,
    grad_faces_from_centers,
    GRID_SPACING,
)
from nucleus.physics.sdf import band_mask
from nucleus.physics.mass_transfer import continuity
from nucleus.physics.ansatz import temperature_ansatz
from nucleus.utils.losses import sdf_sign_bce_loss

from ._api import register_model

__all__ = [
    "Nucleus2MoEDivFree",
    "Nucleus2MoEDivFreeConfig",
    "Nucleus2MoEDivFreeInput",
    "Nucleus2MoEDivFreeOutput",
]


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
class Nucleus2MoEDivFreeInput:
    """The six-channel field container ``step()`` consumes. psi/phi are not part of
    it -- the embedding only reads sdf/temperature/velx/vely (see
    :func:`divfree_input_to_cells`); potentials are model outputs, reconstructed
    downstream, never inputs.

    Shapes (leading batch/time dims written ``...``):
        ``sdf``, ``temperature``: ``(..., H, W)`` cell-centered.
        ``velx``: ``(..., H, W+1)`` x-face velocity.
        ``vely``: ``(..., H+1, W)`` y-face velocity.
    """
    sdf: torch.Tensor
    temperature: torch.Tensor
    velx: torch.Tensor
    vely: torch.Tensor


@dataclass
class Nucleus2MoEDivFreeOutput:
    """Prediction of :class:`Nucleus2MoEDivFree`, each field on its natural grid.

    Shapes match :class:`Nucleus2MoEDivFreeInput`; all fields are in normalized
    units. ``moe_outputs`` is the per-block MoE routing output used for the
    auxiliary / router losses.
    """
    sdf: torch.Tensor
    temperature: torch.Tensor
    velx: torch.Tensor
    vely: torch.Tensor
    velx_sol: torch.Tensor
    vely_sol: torch.Tensor
    velx_dil: torch.Tensor
    vely_dil: torch.Tensor
    phi: torch.Tensor # grad(phi) is component of velocity contributed divergence
    div_source: torch.Tensor # cell-centered divergence source, RHS of poisson solve for phi
    gated_div_source: torch.Tensor # div_source, but with zero gating applied away from interfaces
    moe_outputs: list

    def to_cell_tensor(self) -> torch.Tensor:
        # psi is no longer a direct head output -- reconstruct it from the final
        # velocity so the cell tensor stays comparable to the dataset's/input's.
        psi = stream_function_from_faces(self.velx, self.vely, GRADIENT_SPACING, GRADIENT_SPACING)
        return fields_to_cells(
            self.sdf, self.temperature, self.velx, self.vely, psi, self.phi
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

# NOTE: These settings are hard-coded for pool boiling.
# NOTE: They should ideally be read from the data config files.
# Physical extent of the simulation domain on a cell-centered grid (endpoints
# excluded): height (y) in [0, 16], width (x) in [-8, 8].
DOMAIN_Y_MIN, DOMAIN_Y_MAX = 0.0, 16.0
DOMAIN_X_MIN, DOMAIN_X_MAX = -8.0, 8.0

# Grid spacing passed to the velocity gradients. TODO: fixed-resolution value;
# switch to (extent / num_cells) per axis for a resolution-consistent velocity.
GRADIENT_SPACING = GRID_SPACING


def domain_x_coords(width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Cell-center x positions ``(width,)`` for a sample assumed to start at the
    domain's left edge, spaced by ``GRADIENT_SPACING``. Used where a caller (the
    autoregressive rollout, training/validation steps) has no per-sample x_coords of
    its own to feed the temperature ansatz's heater band.
    """
    return DOMAIN_X_MIN + (torch.arange(width, device=device, dtype=dtype) + 0.5) * GRADIENT_SPACING


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


def clean_phi(phi, div_gate):
    """
    `phi` produced by a model may be noisy / incorrect. This means it can have undesired
    divergence. This function uses div_gate to remove unwanted divergence from phi, while preserving the
    fact that grad(phi) = vel.
    """
    grad_x, grad_y = grad_faces_from_centers(phi, GRADIENT_SPACING, GRADIENT_SPACING)
    div = divergence_centers_from_faces(grad_x, grad_y, GRADIENT_SPACING, GRADIENT_SPACING)
    gated_div = div_gate * div
    return solve_poisson_neumann_dirichlet(gated_div, GRADIENT_SPACING, GRADIENT_SPACING)


def divfree_input_to_cells(
    sdf: torch.Tensor,
    temperature: torch.Tensor,
    velx: torch.Tensor,
    vely: torch.Tensor,
) -> torch.Tensor:
    """Cell-centered input channels the patch embedding consumes: sdf, temperature,
    and the face velocities split onto their two bordering cells. psi/phi are model
    *outputs* only -- they are reconstructed from the predicted potentials and are not
    fed back into the embedding.

    Args (leading batch/time dims written ``...``):
        sdf, temperature: ``(..., H, W)`` cell-centered.
        velx: ``(..., H, W+1)`` x-face velocity.
        vely: ``(..., H+1, W)`` y-face velocity.

    Returns:
        ``(..., H, W, 6)`` ordered ``[sdf, temperature, vel_left, vel_right,
        vel_bottom, vel_top]``.
    """
    vel_left, vel_right = velx[..., :, :-1], velx[..., :, 1:]
    vel_bottom, vel_top = vely[..., :-1, :], vely[..., 1:, :]
    return torch.stack(
        (sdf, temperature, vel_left, vel_right, vel_bottom, vel_top), dim=-1
    )


def fields_to_cells(
    sdf: torch.Tensor,
    temperature: torch.Tensor,
    velx: torch.Tensor,
    vely: torch.Tensor,
    psi: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Split natural-grid divfree fields onto the dataset's 11 cell-centered
    channels: each x-face velocity onto the left/right face of its two cells, each
    y-face velocity onto their bottom/top face, and ``psi`` onto its four cell
    corners. The inverse of the dataset's per-cell split.

    Args (leading batch/time dims written ``...``):
        sdf, temperature, phi: ``(..., H, W)`` cell-centered.
        velx: ``(..., H, W+1)`` x-face velocity.
        vely: ``(..., H+1, W)`` y-face velocity.
        psi: ``(..., H+1, W+1)`` nodal streamfunction.

    Returns:
        ``(..., H, W, 11)`` in the channel order the dataset produces
        ``[sdf, temperature, vel_left, vel_right, vel_bottom, vel_top,
        psi_bl, psi_br, psi_tl, psi_tr, phi]``.
    """
    vel_left, vel_right = velx[..., :, :-1], velx[..., :, 1:]
    vel_bottom, vel_top = vely[..., :-1, :], vely[..., 1:, :]
    return torch.stack(
        (
            sdf, temperature,
            vel_left, vel_right, vel_bottom, vel_top,
            psi[..., :-1, :-1], psi[..., :-1, 1:], psi[..., 1:, :-1], psi[..., 1:, 1:],
            phi,
        ),
        dim=-1,
    )


@register_model("nucleus2_moe_divfree")
@torch.compile(fullgraph=True)
class Nucleus2MoEDivFree(nn.Module):
    config_class = Nucleus2MoEDivFreeConfig
    config_from_dict = staticmethod(_config_from_dict)
    expected_fluid_params = [
        "inv_reynolds", "cpgas", "mugas", "rhogas", "thcogas",
        "stefan", "prandtl", "bulk_temp", "sat_temp"
    ]
    expected_heater_params = ["wallTemp", "xMin", "xMax"]
    expected_global_params = ["gravy"]
    # Cell-centered channels the patch embedding consumes: sdf, temperature, and the
    # x/y-face velocities split onto their two bordering cells.
    expected_fields = [
        "dfun", "temperature",
        "vel_left", "vel_right", "vel_bottom", "vel_top",
    ]
    num_sim_params = len(expected_fluid_params) + len(expected_heater_params) + len(expected_global_params)
    layout = "t h w c"

    def __init__(self, config: Nucleus2MoEDivFreeConfig):
        super().__init__()
        self.config = config
        n_fields = len(self.expected_fields)

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

    def forward(self, batch: DivFreeBatch, normalizer) -> "Nucleus2MoEDivFreeOutput":
        width = batch.input.sdf.shape[-1]
        x_coords = domain_x_coords(width, device=batch.input.sdf.device)
        return self.step(batch, normalizer=normalizer, x_coords=x_coords)

    def model_step(self, input, sim_params: torch.Tensor):
        cells = divfree_input_to_cells(
            input.sdf, input.temperature, input.velx, input.vely
        )
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

    @staticmethod
    def _batched_sim_param(
        sim_params_dict: List[dict], *keys: str, device, dtype=torch.float32
    ) -> torch.Tensor:
        """Stack one scalar sim-parameter across the batch into a shape ``(B,)``
        tensor, e.g. ``_batched_sim_param(dicts, "heater", "wallTemp", device=...)``.
        Nested keys index progressively into each sample's own dict, so every batch
        element gets its own value rather than one dict applied to the whole batch."""
        values = []
        for sample in sim_params_dict:
            value = sample
            for key in keys:
                value = value[key]
            values.append(value)
        return torch.tensor(values, device=device, dtype=dtype)

    def step(
        self,
        input,
        sim_params: torch.Tensor = None,
        normalizer=None,
        x_coords: torch.Tensor = None,
        use_sdf_reinit: bool = False,
        use_leray: bool = False,
        use_div_gate: bool = False,
        use_mass_transfer: bool = False,
        sim_params_dict: Optional[List[dict]] = None,
    ):
        # Accept either a DivFreeBatch (which carries the input fields and the
        # sim-parameter tensor) or a bare field container -- Nucleus2MoEDivFreeInput
        # or DivFreeData -- with sim_params passed separately. Both field containers
        # expose the same six natural-grid fields.
        if isinstance(input, DivFreeBatch):
            sim_params = input.sim_params_tensor
            # The physical sim-parameter dicts continuity/the temperature ansatz need,
            # one per batch element, if not passed explicitly.
            if sim_params_dict is None and input.sim_params:
                sim_params_dict = input.sim_params
            input = input.input
        assert sim_params.dtype == torch.float32, f"expected float32, got {sim_params.dtype}"

        x, moe_outputs = self.model_step(input, sim_params)
        nn_sdf, nn_temp, nn_velx_df, nn_vely_df, nn_div_source = x.unbind(-1)
        # The debed predicts velocity cell-centered (H, W); the reconstruction (leray,
        # grad(phi), output) lives on the MAC faces, so interpolate to faces here.
        # NOTE: placeholder bridge -- revisit whether velocity should be predicted on
        # faces directly or the whole pipeline moved to cell centers.
        nn_velx_df = cells_to_x_face(nn_velx_df)
        nn_vely_df = cells_to_y_face(nn_vely_df)

        nn_sdf_physical = normalizer.unnormalize_sdf(nn_sdf)
        if use_sdf_reinit:
            sdf_physical = sdf_reinit_sussman(nn_sdf_physical, GRADIENT_SPACING, n_iter=5, near_threshold=0.1)
        else:
            sdf_physical = nn_sdf_physical
        sdf = normalizer.normalize_sdf(sdf_physical)

        assert x_coords is not None and sim_params_dict is not None

        # [B, 1, 1, 1]
        batched = lambda *keys: self._batched_sim_param(
            sim_params_dict, *keys, device=nn_temp.device
        )[:, None, None, None]
        # temperature_ansatz adds nn directly to sat_temp/heater_temp (field =
        # sat_temp + decay * nn), so those two need to be in nn's own (normalized)
        # units, not the physical units sim_params_dict carries them in -- normalize
        # them here rather than unnormalizing nn, so temp stays normalized throughout
        # (matching every other Output field) with no extra round-trip.
        bulk_temp_1d = self._batched_sim_param(sim_params_dict, "bulk_temp", device=nn_temp.device)
        sat_temp = normalizer.normalize_temp(batched("sat_temp"), bulk_temp_1d)
        heater_temp = normalizer.normalize_temp(batched("heater", "wallTemp"), bulk_temp_1d)
        temp = temperature_ansatz(
            nn_temp,
            sdf_physical.detach(),
            sat_temp,
            band_width=2 * GRID_SPACING,
            heater_temperature=heater_temp,
            x_coords=x_coords,
            heater_x_min=batched("heater", "xMin"),
            heater_x_max=batched("heater", "xMax"),
            heater_band_width=2 * GRID_SPACING,
        )

        # Construct the divergence-free component of the velocity.
        if use_leray:
            nn_velface_x_physical = normalizer.unnormalize_velx(nn_velx_df)
            nn_velface_y_physical = normalizer.unnormalize_vely(nn_vely_df)
            velfacex_sol_physical, velfacey_sol_physical = leray_projection(
                nn_velface_x_physical, nn_velface_y_physical, GRID_SPACING, GRID_SPACING)
            velfacex_sol = normalizer.normalize_velx(velfacex_sol_physical)
            velfacey_sol = normalizer.normalize_vely(velfacey_sol_physical)
        else:
            velfacex_sol = nn_velx_df
            velfacey_sol = nn_vely_df


        # Find the RHS of the continuity equation: (mass_transfer * dot(n, grad(rho)))
        if use_mass_transfer:
            assert sim_params_dict is not None, "use_mass_transfer requires sim_params_dict"
            div_source_physical = self._continuity_div_source(
                temp.to(torch.float64), sdf_physical.to(torch.float64), sim_params_dict, normalizer)
        else:
            # nn_div_source is the network's normalized raw output; div_source's
            # physical<->normalized scale is vel_std (see the normalize below), not
            # velx's own mean/std -- a divergence source isn't a velocity component
            # and has no reason to carry velx_mean's additive offset.
            div_source_physical = nn_div_source * normalizer.vel_std

        if use_div_gate:
            div_gate = band_mask(sdf_physical, 3.0 * GRADIENT_SPACING).to(self.debed_dtype)
            gated_div_source_physical = div_gate * div_source_physical
        else:
            gated_div_source_physical = div_source_physical
            
        # Normalize the div source and gated div source
        div_source = div_source_physical / normalizer.vel_std
        gated_div_source = gated_div_source_physical / normalizer.vel_std

        phi_physical = solve_poisson_neumann_dirichlet(gated_div_source_physical.to(torch.float64), GRADIENT_SPACING, GRADIENT_SPACING)
        velfacex_dil_physical, velfacey_dil_physical = grad_faces_from_centers(phi_physical, GRADIENT_SPACING, GRADIENT_SPACING)

        phi = normalizer.normalize_phi(phi_physical)
        velfacex_dil = normalizer.normalize_velx(velfacex_dil_physical).to(torch.float32)
        velfacey_dil = normalizer.normalize_vely(velfacey_dil_physical).to(torch.float32)

        velfacex = velfacex_sol + velfacex_dil
        velfacey = velfacey_sol + velfacey_dil

        return Nucleus2MoEDivFreeOutput(
            sdf=sdf.to(torch.float32),
            temperature=temp.to(torch.float32),
            velx=velfacex.to(torch.float32),
            vely=velfacey.to(torch.float32),
            velx_sol=velfacex_sol.to(torch.float32),
            vely_sol=velfacey_sol.to(torch.float32),
            velx_dil=velfacex_dil.to(torch.float32),
            vely_dil=velfacey_dil.to(torch.float32),
            phi=phi.to(torch.float32),
            div_source=div_source.to(torch.float32),
            gated_div_source=gated_div_source.to(torch.float32),
            moe_outputs=moe_outputs,
        )

    def _continuity_div_source(self, temp, sdf_physical, sim_params_dict: List[dict], normalizer):
        device = temp.device
        batched = lambda *keys: self._batched_sim_param(
            sim_params_dict, *keys, device=device, dtype=torch.float64
        )
        bcast = lambda t: t[:, None, None, None]  # (B,) -> (B, 1, 1, 1)

        bulk_temp = batched("bulk_temp")
        heater_temp = batched("heater", "wallTemp")
        scale = heater_temp - bulk_temp
        # non-dimensionalize temperatures; unnormalize_temp broadcasts a (B,) bulk_temp
        # against (B, T, H, W) itself -- the rest of the arithmetic needs it reshaped.
        temp_nd = (normalizer.unnormalize_temp(temp, bulk_temp) - bcast(bulk_temp)) / bcast(scale)
        sat_temp_nd = (batched("sat_temp") - bulk_temp) / scale
        return continuity(
            temp_nd, sdf_physical,
            sat_temp=bcast(sat_temp_nd),
            dx=GRADIENT_SPACING, dy=GRADIENT_SPACING,
            stefan=bcast(batched("stefan")),
            reynolds=1.0 / bcast(batched("inv_reynolds")),
            prandtl=bcast(batched("prandtl")),
            thermal_conductivity=bcast(batched("thcogas")),
            rhogas=bcast(batched("rhogas")),
            wall_temp=1.0,  # heater temperature on the non-dimensional scale
        )

    def _normalized_sim_params(self, sim_params_dict: dict, normalizer, device, batch_size: int) -> torch.Tensor:
        # Assemble the normalized conditioning tensor the network expects from the
        # physical sim-parameter dict, ordered to match the expected_* fields and
        # broadcast to the rollout batch size.
        normalized = normalizer.normalize_params([sim_params_dict])[0]
        values = (
            [normalized[param] for param in self.expected_fluid_params]
            + [normalized["heater"][param] for param in self.expected_heater_params]
            + [normalized[param] for param in self.expected_global_params]
        )
        sim_params = torch.tensor(values, device=device, dtype=torch.float32)
        return sim_params[None, :].expand(batch_size, -1).contiguous()

    def _trajectory_window_to_input(self, window: Trajectory, normalizer, bulk_temp) -> "Nucleus2MoEDivFreeInput":
        return Nucleus2MoEDivFreeInput(
            sdf=normalizer.normalize_sdf(window.sdf),
            temperature=normalizer.normalize_temp(window.temp, bulk_temp),
            velx=normalizer.normalize_velx(window.velx),
            vely=normalizer.normalize_vely(window.vely),
        )

    def _predicted_fields(self, output: "Nucleus2MoEDivFreeOutput", normalizer, bulk_temp, output_time_window_size):
        # Unnormalize the model output to physical units on its natural grids and
        # keep the last output_time_window_size frames of every tracked field.
        keep = slice(-output_time_window_size, None)
        return (
            normalizer.unnormalize_sdf(output.sdf)[:, keep],
            normalizer.unnormalize_temp(output.temperature, bulk_temp)[:, keep],
            normalizer.unnormalize_velx(output.velx)[:, keep],
            normalizer.unnormalize_vely(output.vely)[:, keep],
        )

    def forward_trajectory(
        self,
        trajectory: Trajectory,
        normalizer,
        dx: float,
        input_time_window_size: int,
        output_time_window_size: int,
        trajectory_steps: int,
        use_sdf_reinit: bool = False,
        # forward_trajectory is the inference/rollout entry point (never used for the
        # training loss, which calls step() directly), so it's the right place to
        # enforce Leray projection for a divergence-free rollout even though training
        # only encourages it approximately via the loss (see _auxiliary_targets).
        # Left off by default for now while debugging -- pass use_leray=True to enable.
        use_leray: bool = False,
        use_div_gate: bool = False,
        use_mass_transfer: bool = False,
        return_moe_outputs: bool = False,
        clip_temp: bool = False,
    ) -> Trajectory:
        assert input_time_window_size <= trajectory.num_steps

        # forward_trajectory always runs a single physical simulation (one Trajectory,
        # one sim_params dict), unlike step()'s batch of possibly-different samples --
        # so bulk_temp/the conditioning tensor/clip_temp use that one dict directly,
        # while sim_params_dict passed into step() is still the (length-1) list its
        # per-batch-element contract expects.
        sim_params_dict = trajectory.sim_params[0]
        bulk_temp = sim_params_dict["bulk_temp"]
        sim_params = self._normalized_sim_params(
            sim_params_dict, normalizer, trajectory.sdf.device, trajectory.sdf.shape[0]
        )
        x_coords = domain_x_coords(trajectory.sdf.shape[-1], device=trajectory.sdf.device)

        trajectory_moe_outputs = [] if return_moe_outputs else None

        for _ in range(input_time_window_size, trajectory_steps, output_time_window_size):
            model_input = self._trajectory_window_to_input(
                trajectory.last(input_time_window_size), normalizer, bulk_temp
            )
            output = self.step(
                model_input, sim_params, normalizer, x_coords=x_coords,
                use_sdf_reinit=use_sdf_reinit, use_leray=use_leray, use_div_gate=use_div_gate,
                use_mass_transfer=use_mass_transfer, sim_params_dict=trajectory.sim_params,
            )
            sdf, temp, velx, vely = self._predicted_fields(
                output, normalizer, bulk_temp, output_time_window_size
            )

            if clip_temp:
                temp = clip_temp_by_phase(
                    temp, sdf,
                    sim_params_dict["sat_temp"],
                    sim_params_dict["heater"]["wallTemp"],
                )

            trajectory = trajectory.extend(sdf, temp, velx, vely)
            if return_moe_outputs:
                trajectory_moe_outputs.append(output.moe_outputs)

        if return_moe_outputs:
            return trajectory, trajectory_moe_outputs
        return trajectory


@dataclass
class AuxiliaryTargets:
    """Targets derived from the dataset's base DivFreeData target data."""
    velx_sol: torch.Tensor
    vely_sol: torch.Tensor
    velx_dil: torch.Tensor
    vely_dil: torch.Tensor


class DivFreeForecastModule(MoEConditionedForecastModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Override ModuleBase's augmentations: each divfree field is augmented on its
        # own natural grid as a single channel (see _augment_field), so only additive
        # noise applies for now
        self.augmentations = [LogUniformNoise(0.001, 5e-2, skip_prob=0.1)]

    # How often to log the per-head gradient/weight diagnostics. Scalars are cheap;
    # histograms are heavier, so they go at a coarser cadence.
    DEBED_SCALAR_LOG_INTERVAL = 100
    DEBED_HISTOGRAM_LOG_INTERVAL = 500

    def _sdf_sign_loss(self, output: Nucleus2MoEDivFreeOutput, target: DivFreeData) -> torch.Tensor:
        # penalize incorrect signs in the output SDF. Near the interface, an incorrect sign may result
        # in a small L1 error, and not be learned well. However, predicing phase incorrectly is extremely
        # problematic and so it need to be penalized heavily.
        pred_sdf_physical = self.normalizer.unnormalize_sdf(output.sdf)
        target_sdf_physical = self.normalizer.unnormalize_sdf(target.sdf)
        
        num_vapor = (target_sdf_physical > 0).sum()
        num_liquid = target_sdf_physical.numel() - num_vapor
        vapor_weight = (num_liquid / num_vapor).item() if num_vapor > 0 else 1.0

        return sdf_sign_bce_loss(pred_sdf_physical, target_sdf_physical, vapor_weight)

    def _debed_heads(self) -> Dict[str, nn.Module]:
        """The output head(s) whose gradients/weights are logged separately."""
        return {
            "debed": self.model.debed,
        }

    def _wandb_run(self):
        """The active wandb run if a WandbLogger is attached, else None. Histograms
        go through the run directly since Lightning's self.log only takes scalars."""
        logger = self.logger
        if isinstance(logger, WandbLogger):
            return logger.experiment
        return None

    def _debed_diagnostics(self, include_scalars: bool, include_histograms: bool):
        """Per-head gradient and weight diagnostics as (scalar_log, histogram_log).
        Grad stats span every parameter of the head; the weight distribution is the
        >=2D weight tensors only (excludes 1D biases)."""
        scalar_log: dict = {}
        histogram_log: dict = {}
        for name, head in self._debed_heads().items():
            grads = [p.grad.detach().flatten() for p in head.parameters() if p.grad is not None]
            weights = [p.detach().flatten() for p in head.parameters() if p.ndim >= 2]

            if grads:
                grad_vector = torch.cat(grads)
                if include_scalars:
                    scalar_log[f"train/{name}/grad_norm"] = grad_vector.norm()
                    scalar_log[f"train/{name}/grad_absmax"] = grad_vector.abs().max()
                if include_histograms:
                    histogram_log[f"train/{name}/grad_hist"] = wandb.Histogram(
                        grad_vector.float().cpu().numpy()
                    )
            if weights:
                weight_vector = torch.cat(weights)
                if include_scalars:
                    scalar_log[f"train/{name}/weight_std"] = weight_vector.std()
                    scalar_log[f"train/{name}/weight_absmax"] = weight_vector.abs().max()
                if include_histograms:
                    histogram_log[f"train/{name}/weight_hist"] = wandb.Histogram(
                        weight_vector.float().cpu().numpy()
                    )
        return scalar_log, histogram_log

    def on_before_optimizer_step(self, optimizer):
        # Gradients are populated here (after backward, before the optimizer step).
        super().on_before_optimizer_step(optimizer)
        if not self.trainer.is_global_zero:
            return

        log_scalars = self.global_step % self.DEBED_SCALAR_LOG_INTERVAL == 0
        run = self._wandb_run()
        log_histograms = run is not None and self.global_step % self.DEBED_HISTOGRAM_LOG_INTERVAL == 0
        if not (log_scalars or log_histograms):
            return

        scalar_log, histogram_log = self._debed_diagnostics(log_scalars, log_histograms)
        if scalar_log:
            # Match the base class's grad-norm log: per-step, no epoch reduction. Without
            # these, self.log applies its non-training-step defaults (on_epoch=True) and
            # the scalars get bucketed to epoch boundaries instead of logged per step.
            self.default_log_dict(scalar_log, on_step=True, on_epoch=False)
        if histogram_log:
            # commit=False (and no explicit step) appends the histograms to wandb's
            # current step without advancing its counter. Passing step=global_step here
            # instead races Lightning's own step-committed logging: it jumps wandb ahead
            # and drops the metrics Lightning had buffered for the preceding steps.
            run.log(histogram_log, commit=False)

    @staticmethod
    def _model_input(data: DivFreeData) -> Nucleus2MoEDivFreeInput:
        return Nucleus2MoEDivFreeInput(
            sdf=data.sdf, temperature=data.temperature, velx=data.velx, vely=data.vely,
        )

    @staticmethod
    def _data_cell_tensor(data: DivFreeData) -> torch.Tensor:
        # Ground-truth cell tensor for plotting/logging -- unlike Nucleus2MoEDivFreeOutput
        # (whose psi is reconstructed from velocity), this uses the dataset's own psi/phi.
        return fields_to_cells(
            data.sdf, data.temperature, data.velx, data.vely, data.psi, data.phi
        )

    def _auxiliary_targets(self, target: DivFreeData) -> AuxiliaryTargets:
        velx_physical = self.normalizer.unnormalize_velx(target.velx)
        vely_physical = self.normalizer.unnormalize_vely(target.vely)
        velx_sol_physical, vely_sol_physical = leray_projection(
            velx_physical, vely_physical, GRADIENT_SPACING, GRADIENT_SPACING
        )
        velx_dil_physical = velx_physical - velx_sol_physical
        vely_dil_physical = vely_physical - vely_sol_physical
        return AuxiliaryTargets(
            velx_sol=self.normalizer.normalize_velx(velx_sol_physical),
            vely_sol=self.normalizer.normalize_vely(vely_sol_physical),
            velx_dil=self.normalizer.normalize_velx(velx_dil_physical),
            vely_dil=self.normalizer.normalize_vely(vely_dil_physical),
        )

    def _per_field_mae(self, output: Nucleus2MoEDivFreeOutput, target: DivFreeData, prefix: str) -> dict:
        log_field_names = ["sdf", "temperature", "velx", "vely"]
        metrics = {
            f"{prefix}/mae_{name}": (getattr(output, name) - getattr(target, name)).abs().mean()
            for name in log_field_names
        }
        auxiliary_targets = self._auxiliary_targets(target)
        metrics[f"{prefix}/mae_velx_sol"] = (output.velx_sol - auxiliary_targets.velx_sol).abs().mean()
        metrics[f"{prefix}/mae_vely_sol"] = (output.vely_sol - auxiliary_targets.vely_sol).abs().mean()
        metrics[f"{prefix}/mae_velx_dil"] = (output.velx_dil - auxiliary_targets.velx_dil).abs().mean()
        metrics[f"{prefix}/mae_vely_dil"] = (output.vely_dil - auxiliary_targets.vely_dil).abs().mean()
        metrics[f"{prefix}/mae_div_source"] = (
            output.div_source - self._target_div_source(target)
        ).abs().mean()
        return metrics

    def _target_div_source(self, target: DivFreeData) -> torch.Tensor:
        # The divergence of the normalized velocity is the physical
        # divergence normalized by the vel_std (see DivFreeNormalizer.)
        return divergence_centers_from_faces(
            target.velx, target.vely, GRADIENT_SPACING, GRADIENT_SPACING
        )

    def _loss_fields(self, output: Nucleus2MoEDivFreeOutput, target: DivFreeData):
        names = ["sdf", "temperature", "velx", "vely"]
        pred_grids = [getattr(output, name) for name in names]
        target_grids = [getattr(target, name) for name in names]

        auxiliary_targets = self._auxiliary_targets(target)
        pred_grids += [output.velx_sol, output.vely_sol, output.velx_dil, output.vely_dil]
        target_grids += [
            auxiliary_targets.velx_sol, auxiliary_targets.vely_sol,
            auxiliary_targets.velx_dil, auxiliary_targets.vely_dil,
        ]

        # The divergence source has no field on the DivFreeData target
        pred_grids.append(output.div_source)
        target_grids.append(self._target_div_source(target))
        return pred_grids, target_grids

    def _field_loss(self, output: Nucleus2MoEDivFreeOutput, target: DivFreeData) -> torch.Tensor:
        pred_grids, target_grids = self._loss_fields(output, target)
        total_abs_error = sum(
            (pred_grid - target_grid).abs().sum()
            for pred_grid, target_grid in zip(pred_grids, target_grids)
        )
        total_elements = sum(pred_grid.numel() for pred_grid in pred_grids)
        return total_abs_error / total_elements

    def _augment_field(self, field: torch.Tensor) -> torch.Tensor:
        augmented = field.unsqueeze(-1)
        for augmentation in self.augmentations:
            augmented = augmentation(augmented)
        return augmented.squeeze(-1)

    def _augment(self, data: DivFreeData) -> DivFreeData:
        # Apply the augmentations to every input field on its natural grid.
        return data._apply(self._augment_field)

    def training_step(self, batch: DivFreeBatch, batch_idx: int) -> torch.Tensor:
        torch.compiler.cudagraph_mark_step_begin()
        model_input = self._model_input(self._augment(batch.input))
        x_coords = domain_x_coords(batch.input.sdf.shape[-1], device=batch.input.sdf.device)
        output = self.model.step(
            model_input, batch.sim_params_tensor, self.normalizer,
            x_coords=x_coords, sim_params_dict=batch.sim_params,
        )

        data_loss = self._field_loss(output, batch.target)
        sdf_sign_loss = self._sdf_sign_loss(output, batch.target)

        aux_loss, router_has_loss = self._router_loss(output.moe_outputs)
        loss = data_loss + sdf_sign_loss + aux_loss
        self._update_router_bias(output.moe_outputs)

        log_dict = {
            "train/loss": loss,
            "train/data_loss": data_loss,
            "train/sdf_sign_loss": sdf_sign_loss,
            "train/step": self.global_step,
            "train/learning_rate": self.get_current_lr(),
        }
        log_dict |= self._per_field_mae(output, batch.target, "train")
        log_dict = self._moe_metrics(output.moe_outputs, log_dict, "train")
        self.default_log_dict(log_dict)
        return loss

    def validation_step(self, batch: DivFreeBatch, batch_idx: int) -> torch.Tensor:
        x_coords = domain_x_coords(batch.input.sdf.shape[-1], device=batch.input.sdf.device)
        output = self.model.step(
            self._model_input(batch.input), batch.sim_params_tensor, self.normalizer,
            x_coords=x_coords, sim_params_dict=batch.sim_params,
        )
        data_loss = self._field_loss(output, batch.target)
        sdf_sign_loss = self._sdf_sign_loss(output, batch.target)
        loss = data_loss + sdf_sign_loss

        # Re-stack onto the dataset's cell channels for plotting / tensor metrics.
        pred_cells = output.to_cell_tensor()
        target_cells = self._data_cell_tensor(batch.target)
        if batch_idx == 0:
            input_cells = self._data_cell_tensor(batch.input)
            self.validation_sample = (input_cells.detach(), target_cells.detach(), pred_cells.detach())

        log_dict = {"val/loss": loss, "val/data_loss": data_loss, "val/sdf_sign_loss": sdf_sign_loss}
        log_dict |= self._per_field_mae(output, batch.target, "val")
        log_dict = self.log_step_metrics(
            log_dict, pred_cells, target_cells, batch.dx[0].item(), batch.dy[0].item(), "val"
        )
        log_dict = self._moe_metrics(output.moe_outputs, log_dict, "val")
        self.default_log_dict(log_dict)
        return loss
