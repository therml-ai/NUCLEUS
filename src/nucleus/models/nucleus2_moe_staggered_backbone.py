"""Layers shared by the staggered-grid NUCLEUS-2 models. Subclasses define their own
step, rollout and training module. Parameter names are part of the checkpoint format."""
import dataclasses
from dataclasses import dataclass
from typing import List, Literal

import torch
import torch.nn as nn
from torch.profiler import record_function
from rotary_embedding_torch import RotaryEmbedding

from nucleus.layers import AdaptiveEmbed, AdaptiveDebed, LinearEmbed, LinearDebed
from nucleus.layers.adaptive_layernorm import AdaptiveLayerNorm
from nucleus.layers.attention import NeighborhoodAttention
from nucleus.layers.droppath import DropPath
from nucleus.layers.moe.topk_moe import TopkMoE, TopkMoEOutput, TopkRouterWithBias

__all__ = [
    "NUCLEUS_DTYPE",
    "Nucleus2MoEStaggeredBackbone",
    "Nucleus2MoEStaggeredBackboneConfig",
    "StaggeredFields",
    "TransformerMoEBlock",
    "faces_from_cell_split",
    "get_dtype",
    "staggered_to_cells",
]

NUCLEUS_DTYPE = Literal["float32", "float16", "bfloat16"]

_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
_STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}
_DTYPE_FIELDS = ("embed_dtype", "debed_dtype", "activation_dtype", "attention_dtype", "moe_dtype")


def get_dtype(dtype):
    if isinstance(dtype, str):
        return _STR_TO_DTYPE[dtype]
    return dtype


@dataclass
class Nucleus2MoEStaggeredBackboneConfig:
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


def config_to_dict(config) -> dict:
    """Serializable dict of any backbone config (or subclass); dtypes become strings."""
    d = dataclasses.asdict(config)
    for key in _DTYPE_FIELDS:
        if not isinstance(d[key], str):
            d[key] = _DTYPE_TO_STR[d[key]]
    return d


def config_from_dict(config_class, d: dict):
    """Inverse of :func:`config_to_dict` for ``config_class``."""
    d = dict(d)
    for key in _DTYPE_FIELDS:
        if not isinstance(d[key], torch.dtype):
            d[key] = _STR_TO_DTYPE[d[key]]
    return config_class(**d)


@dataclass
class StaggeredFields:
    """The four fields on their natural grids (leading batch/time dims written ``...``):
    ``sdf``, ``temperature`` ``(..., H, W)``; ``velx`` ``(..., H, W+1)``;
    ``vely`` ``(..., H+1, W)``."""
    sdf: torch.Tensor
    temperature: torch.Tensor
    velx: torch.Tensor
    vely: torch.Tensor


def staggered_to_cells(
    sdf: torch.Tensor,
    temperature: torch.Tensor,
    velx: torch.Tensor,
    vely: torch.Tensor,
) -> torch.Tensor:
    """Cell-centered channels for the patch embedding: sdf, temperature, and each face
    velocity copied onto its two bordering cells.

    Returns ``(..., H, W, 6)`` ordered ``[sdf, temperature, vel_left, vel_right,
    vel_bottom, vel_top]``.
    """
    vel_left, vel_right = velx[..., :, :-1], velx[..., :, 1:]
    vel_bottom, vel_top = vely[..., :-1, :], vely[..., 1:, :]
    return torch.stack((sdf, temperature, vel_left, vel_right, vel_bottom, vel_top), dim=-1)


def faces_from_cell_split(
    vel_left: torch.Tensor,
    vel_right: torch.Tensor,
    vel_bottom: torch.Tensor,
    vel_top: torch.Tensor,
):
    """Inverse of the per-cell face split in :func:`staggered_to_cells`.

    Each interior face appears twice, as the right/top face of one cell and the
    left/bottom face of its neighbor; its value is the average of the two. Wall faces
    have one owner and take its value.

    Args (leading dims written ``...``): four ``(..., H, W)`` tensors.

    Returns:
        ``velx`` ``(..., H, W+1)`` and ``vely`` ``(..., H+1, W)``.
    """
    velx = torch.cat(
        (
            vel_left[..., :, :1],
            0.5 * (vel_right[..., :, :-1] + vel_left[..., :, 1:]),
            vel_right[..., :, -1:],
        ),
        dim=-1,
    )
    vely = torch.cat(
        (
            vel_bottom[..., :1, :],
            0.5 * (vel_top[..., :-1, :] + vel_bottom[..., 1:, :]),
            vel_top[..., -1:, :],
        ),
        dim=-2,
    )
    return velx, vely


class TransformerMoEBlock(nn.Module):
    def __init__(self, config: Nucleus2MoEStaggeredBackboneConfig, num_sim_params: int, drop_path_prob: float):
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
            softmax_first=False,
        )

        self.attention = NeighborhoodAttention(embed_dim=config.embed_dim, num_heads=config.num_heads)

        self.mlp = TopkMoE(
            num_experts=config.num_experts,
            hidden_dim=config.embed_dim,
            intermediate_dim=config.moe_intermediate_dim,
            topk=config.topk,
            router=self.router,
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


class Nucleus2MoEStaggeredBackbone(nn.Module):
    """Layers and token pipeline shared by the staggered-grid NUCLEUS-2 models.

    Subclasses pass ``out_channels`` (the debed head width), set ``config_class``, and
    define their own ``step``/``forward``/``forward_trajectory``. Not registered and not
    compiled; register and compile the subclass.
    """

    config_class = Nucleus2MoEStaggeredBackboneConfig
    expected_fluid_params = [
        "inv_reynolds", "cpgas", "mugas", "rhogas", "thcogas",
        "stefan", "prandtl", "bulk_temp", "sat_temp",
    ]
    expected_heater_params = ["wallTemp", "xMin", "xMax"]
    expected_global_params = ["gravy"]
    # Channels the patch embedding consumes (see staggered_to_cells).
    expected_fields = [
        "dfun", "temperature",
        "vel_left", "vel_right", "vel_bottom", "vel_top",
    ]
    num_sim_params = len(expected_fluid_params) + len(expected_heater_params) + len(expected_global_params)
    layout = "t h w c"

    @classmethod
    def config_from_dict(cls, d: dict):
        return config_from_dict(cls.config_class, d)

    def __init__(self, config: Nucleus2MoEStaggeredBackboneConfig, out_channels: int):
        super().__init__()
        self.config = config
        in_channels = len(self.expected_fields)

        self.embed_dtype = get_dtype(config.embed_dtype)
        self.debed_dtype = get_dtype(config.debed_dtype)

        self.rotary_emb = RotaryEmbedding(
            dim=(config.embed_dim // config.num_heads) // 3,
            freqs_for="pixel",
            max_freq=256,
            seq_before_head_dim=True,
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
                patch_size=config.patch_size, in_channels=in_channels,
                embed_dim=config.embed_dim, dtype=self.embed_dtype,
            )
            self.debed = LinearDebed(
                patch_size=config.patch_size, embed_dim=config.embed_dim,
                out_channels=out_channels, dtype=self.debed_dtype,
            )
        else:
            self.embed = AdaptiveEmbed(
                in_channels=in_channels, out_channels=config.embed_dim,
                out_shape=(16, 16), dtype=config.embed_dtype,
            )
            self.debed = AdaptiveDebed(
                in_channels=config.embed_dim, out_channels=out_channels,
                patch_shape=(16, 16), dtype=config.debed_dtype,
            )

    def get_extra_state(self):
        return {"model_name": getattr(self, "_model_name", None), "config": config_to_dict(self.config)}

    def set_extra_state(self, state):
        self._model_name = state.get("model_name")
        self.config = self.config_from_dict(state["config"])

    def model_step(self, input, sim_params: torch.Tensor):
        """Fields on their natural grids (any container with ``sdf``, ``temperature``,
        ``velx``, ``vely``) and normalized ``sim_params`` ``(B, num_sim_params)`` in;
        debed output ``(B, T, H, W, out_channels)`` and per-block MoE outputs out."""
        cells = staggered_to_cells(input.sdf, input.temperature, input.velx, input.vely)
        assert cells.dtype == torch.float32, f"expected float32, got {cells.dtype}"
        _, _, h, w, _ = cells.shape

        with record_function("encode"):
            x = embed = self.embed(cells.to(self.embed_dtype))
        x = x.to(get_dtype(self.config.activation_dtype))

        with record_function("get_axial_freqs"):
            with torch.no_grad():
                _, embed_t, embed_h, embed_w, _ = embed.shape
                rotary_freqs = self.rotary_emb.get_axial_freqs(embed_t, embed_h, embed_w)[None, :, :, :, None, :]

        moe_outputs: List[TopkMoEOutput] = []
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
                x, moe_output = blk(x, rotary_freqs, sim_params)
                moe_outputs.append(moe_output)

        with record_function("debed"):
            x = self.out_norm(x.to(self.debed_dtype) + embed.to(self.debed_dtype))
            x = self.debed(x, target_shape=(h, w))
        return x, moe_outputs
