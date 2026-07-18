import dataclasses
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.profiler import record_function
from rotary_embedding_torch import RotaryEmbedding
from typing import Literal, Optional

from nucleus.layers.adaptive_layernorm import AdaptiveLayerNorm
from nucleus.layers.attention import NeighborhoodAttention
from nucleus.layers.moe.topk_moe import TopkMoE, TopkMoEOutput, TopkRouterWithBias
from nucleus.layers.droppath import DropPath
from nucleus.layers import (
    AdaptiveEmbed,
    AdaptiveDebed,
    LinearEmbed,
    LinearDebed
)
from nucleus.data.batching import CollatedBatch
from nucleus.utils.sdf_reinit import sdf_reinit_sussman
from nucleus.utils.inf_stabilizer import clip_temp_by_phase

from ._api import register_model

__all__ = ["Nucleus2MoEDivFree", "Nucleus2MoEDivFreeConfig"]


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

class PhaseModulate(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.phase_table = nn.Embedding(2, 2 * embed_dim)
        # Start as identity: gamma = beta = 0 so (1 + gamma) * x + beta == x.
        nn.init.zeros_(self.phase_table.weight)

    def forward(self, x: torch.Tensor, phase_mask: torch.Tensor) -> torch.Tensor:
        assert x.shape[:-1] == phase_mask.shape

        gamma_beta = self.phase_table(phase_mask)  # (B, T, H, W, 2 * C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        assert gamma.shape == x.shape
        assert beta.shape == x.shape

        return (1 + gamma) * x + beta

# NOTE: These settings are hard-coded for pool boiling.
# NOTE: They should ideally be read from the data config files.
# Physical extent of the simulation domain on a cell-centered grid (endpoints
# excluded): height (y) in [0, 16], width (x) in [-8, 8].
DOMAIN_Y_MIN, DOMAIN_Y_MAX = 0.0, 16.0
DOMAIN_X_MIN, DOMAIN_X_MAX = -8.0, 8.0

# Grid spacing passed to the velocity gradients. TODO: fixed-resolution value;
# switch to (extent / num_cells) per axis for a resolution-consistent velocity.
GRADIENT_SPACING = 1 / 32

# Free-slip window length scale, in grid cells: psi is windowed to zero at the
# closed walls over ~this many cells (tanh) so each wall is a streamline
# (v . n = 0, no-penetration) with the tangential velocity left free.
WALL_SLIP_CELLS = 2.0

# Dilatational wall band, in grid cells so it stays a fixed physical-cell
# thickness at any resolution: grad(phi) is held at exactly zero within
# WALL_ZERODIV_CELLS of every closed wall (so it adds no divergence there) and
# ramps back to full over the next WALL_ZERODIV_RAMP_CELLS.
WALL_ZERODIV_CELLS = 2.0
WALL_ZERODIV_RAMP_CELLS = 2.0


def _cell_centers(num_cells, low, high, device, dtype):
    step = (high - low) / num_cells
    return low + (torch.arange(num_cells, device=device, dtype=dtype) + 0.5) * step


def _smoothstep(t):
    # Smooth (C1) 0 -> 1 ramp on [0, 1], clamped outside.
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def vapor_gate_from_sdf(sdf, band):
    # Exactly 0 for sdf <= -band (deep liquid -> exactly div-free), exactly 1 for
    # sdf >= 0 (vapor), with a *smooth* (C1) smoothstep across the band. A linear
    # clamp would kink at both ends of the band, and since velocity = curl(psi) +
    # gate * grad(phi), those kinks imprint as slope-discontinuity rings in the
    # velocity; smoothstep keeps the velocity C1 across the interface.
    return _smoothstep(sdf / band + 1.0)


def free_slip_psi_window(height, width, device, dtype):
    """Streamfunction window for a free-slip boundary: it vanishes on the closed
    walls (bottom/left/right) with a *nonzero* wall-normal slope (tanh), so
    windowing psi by it makes each wall a streamline -- the wall-normal velocity
    goes to zero (no-penetration) while the tangential velocity stays free. The
    top is open (window == 1). curl(window * psi) is still exactly divergence free.
    """
    cell_y = (DOMAIN_Y_MAX - DOMAIN_Y_MIN) / height
    cell_x = (DOMAIN_X_MAX - DOMAIN_X_MIN) / width
    y = _cell_centers(height, DOMAIN_Y_MIN, DOMAIN_Y_MAX, device, dtype)
    x = _cell_centers(width, DOMAIN_X_MIN, DOMAIN_X_MAX, device, dtype)

    ramp_y = torch.tanh((y - DOMAIN_Y_MIN) / (WALL_SLIP_CELLS * cell_y))
    ramp_x = (torch.tanh((x - DOMAIN_X_MIN) / (WALL_SLIP_CELLS * cell_x))
              * torch.tanh((DOMAIN_X_MAX - x) / (WALL_SLIP_CELLS * cell_x)))
    return ramp_y[:, None] * ramp_x[None, :]


def dilatational_wall_mask(height, width, device, dtype):
    """Per-cell mask that is *exactly* 0 within a band of every closed wall
    (bottom/left/right) and 1 in the interior, so the divergent grad(phi) part
    contributes nothing near the walls -- the velocity there is pure curl(psi) and
    hence divergence free. The top is open (mask == 1).
    """
    cell_y = (DOMAIN_Y_MAX - DOMAIN_Y_MIN) / height
    cell_x = (DOMAIN_X_MAX - DOMAIN_X_MIN) / width
    y = _cell_centers(height, DOMAIN_Y_MIN, DOMAIN_Y_MAX, device, dtype)
    x = _cell_centers(width, DOMAIN_X_MIN, DOMAIN_X_MAX, device, dtype)

    # Distance into the domain from each closed wall.
    dist_bottom = y - DOMAIN_Y_MIN
    dist_left = x - DOMAIN_X_MIN
    dist_right = DOMAIN_X_MAX - x

    def interior(distance, cell_size):
        band = WALL_ZERODIV_CELLS * cell_size
        ramp = WALL_ZERODIV_RAMP_CELLS * cell_size
        return _smoothstep((distance - band) / ramp)

    return (interior(dist_bottom, cell_y)[:, None]
            * interior(dist_left, cell_x)[None, :]
            * interior(dist_right, cell_x)[None, :])


def velocity_from_potentials(psi, phi, vapor_gate):
    """Velocity from the streamfunction psi (solenoidal / divergence-free) plus a
    dilatational grad(phi) part gated to vapor and masked to zero in a band around
    every closed wall. psi is windowed to zero at the walls (free-slip: v . n = 0
    with the tangential velocity free), and the dilatational mask (which vanishes
    at the walls too) keeps grad(phi) from adding divergence there. So the velocity
    is exactly divergence free in the deep liquid (gate == 0) and near the walls,
    with divergence only in the vapor interior, and satisfies free-slip at every
    closed wall.
    """
    height, width = psi.shape[-2], psi.shape[-1]
    psi = psi * free_slip_psi_window(height, width, psi.device, psi.dtype)
    dilatational_mask = dilatational_wall_mask(height, width, psi.device, psi.dtype)

    velx_sol = torch.gradient(psi, dim=-2, spacing=GRADIENT_SPACING)[0]     #  ∂ψ/∂y
    vely_sol = -torch.gradient(psi, dim=-1, spacing=GRADIENT_SPACING)[0]    # -∂ψ/∂x
    velx_dil = torch.gradient(phi, dim=-1, spacing=GRADIENT_SPACING)[0]     #  ∂φ/∂x
    vely_dil = torch.gradient(phi, dim=-2, spacing=GRADIENT_SPACING)[0]     #  ∂φ/∂y

    gated_dilatational = vapor_gate * dilatational_mask
    velx = velx_sol + gated_dilatational * velx_dil
    vely = vely_sol + gated_dilatational * vely_dil
    return velx, vely


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
    expected_fields = ["dfun", "temperature", "velx", "vely"]
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
                out_channels=2,
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
                out_channels=2,
                patch_shape=(16, 16),
                dtype=config.debed_dtype
            )

        self.debed_full = LinearDebed(
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            out_channels=2,
            dtype=self.debed_dtype
        )

        # Zero-init the output head so the predicted residuals start at zero, rather
        # than being extremely noisy
        nn.init.zeros_(self.debed.linear.weight)

    def _init_head(self):
        # PyTorch's default kaiming_uniform_(a=sqrt(5)) under-scales every
        # Linear/Conv (std x ~0.58 each), collapsing the head output to ~0.2 std.
        # Re-init with variance-preserving gains so it stays ~1: "linear" gain for
        # the projections, "relu" gain ahead of the GELU to offset its shrinkage.
        nn.init.kaiming_normal_(self.sdf_head.linear.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.field_head.linear.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.field_output[0].conv.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.field_output[2].conv.weight, nonlinearity="linear")
        nn.init.zeros_(self.field_output[0].conv.bias)
        nn.init.zeros_(self.field_output[2].conv.bias)

    def get_extra_state(self):
        return {"model_name": getattr(self, "_model_name", None), "config": _config_to_dict(self.config)}

    def set_extra_state(self, state):
        self._model_name = state.get("model_name")
        self.config = _config_from_dict(state["config"])

    def forward(self, batch: CollatedBatch) -> torch.Tensor:
        return self.step(batch.input, batch.sim_params_tensor)

    def step(self, input: torch.Tensor, sim_params: torch.Tensor):
        assert input.dtype == torch.float32
        assert sim_params.dtype == torch.float32

        _, _, h, w, _ = input.shape

        with record_function("encode"):
            x = embed = self.embed(input.to(self.embed_dtype))

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
            sdf_delta, temp_delta = self.debed(x, target_shape=(h, w)).unbind(-1)
            psi, phi = self.debed_full(x, target_shape=(h, w)).unbind(-1)

            # Residual (persistence) parameterization: the head predicts the change
            # from the input, so a zero-initialized head starts at output == input
            # rather than noise. The velocity residual is curl(psi) + gated grad(phi),
            # so it stays divergence free and free-slip as long as the input is
            # (true of the data and of every div-free rollout step).
            input_sdf, input_temp, _, _ = input.to(self.debed_dtype).unbind(-1)
            sdf = input_sdf + sdf_delta
            temp = input_temp + temp_delta

            vapor_gate = vapor_gate_from_sdf(sdf, band=2.0)
            velx, vely = velocity_from_potentials(psi, phi, vapor_gate)
            x = torch.stack((sdf, temp, velx, vely), dim=-1)

        return x.to(torch.float32), moe_outputs

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

    def forward_trajectory(
        self,
        initial_state: torch.Tensor,
        sim_params_dict: dict,
        normalizer,
        dx: float,
        input_time_window_size: int,
        output_time_window_size: int,
        trajectory_steps: int,
        use_sdf_reinit: bool = False,
        return_moe_outputs: bool = False,
        clip_temp: bool = False,
    ):
        assert initial_state.dim() == 5, "initial state must be [B, T, H, W, C]"
        assert input_time_window_size <= initial_state.shape[1]

        bulk_temp = sim_params_dict["bulk_temp"]
        sim_params = self._normalized_sim_params(
            sim_params_dict, normalizer, initial_state.device, initial_state.shape[0]
        )

        trajectory = initial_state.clone()
        trajectory_moe_outputs = [] if return_moe_outputs else None

        for _ in range(input_time_window_size, trajectory_steps, output_time_window_size):
            normalized_window = normalizer.normalize(trajectory[:, -input_time_window_size:], bulk_temp)
            pred, moe_outputs = self.step(normalized_window, sim_params)
            pred = normalizer.unnormalize(pred, bulk_temp)
            output_time_window = pred[:, -output_time_window_size:]

            if use_sdf_reinit:
                output_time_window[..., 0] = sdf_reinit_sussman(output_time_window[..., 0], dx=dx, n_iter=5)

            if clip_temp:
                output_time_window[..., 1] = clip_temp_by_phase(
                    output_time_window[..., 1],
                    output_time_window[..., 0],
                    sim_params_dict["sat_temp"],
                    sim_params_dict["heater"]["wallTemp"],
                )

            trajectory = torch.cat((trajectory, output_time_window), dim=1)
            if return_moe_outputs:
                trajectory_moe_outputs.append(moe_outputs)

        if return_moe_outputs:
            return trajectory, trajectory_moe_outputs
        return trajectory