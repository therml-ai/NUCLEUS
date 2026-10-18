import dataclasses
from dataclasses import dataclass
from typing import List, Literal, Tuple, Type, TypeVar

import torch
import torch.nn as nn
from torch.profiler import record_function
from rotary_embedding_torch import RotaryEmbedding
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper

from nucleus.layers.adaptive_layernorm import AdaptiveLayerNorm
from nucleus.layers.attention import NeighborhoodAttention
from nucleus.layers.moe.topk_moe import TopkMoE, TopkMoEOutput, TopkRouterWithBias
from nucleus.layers.droppath import DropPath
from nucleus.layers.cast import CastRMSNorm
from nucleus.layers import (
    AdaptiveEmbed,
    AdaptiveDebed,
    LinearEmbed,
    LinearDebed,
)
from nucleus.models.flashx_io import (
    NUCLEUS2_CONFIG_KEYS,
    num_config_columns,
    next_frames,
    restored_config,
    unnormalized_prediction,
)

__all__ = [
    "Nucleus2MoEBase",
    "Nucleus2MoEBaseConfig",
    "TransformerMoEBlock",
    "config_from_dict",
    "config_to_dict",
    "get_dtype",
    "patch_debed",
]

_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
_STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}
_DTYPE_FIELDS = ("embed_dtype", "debed_dtype", "activation_dtype", "attention_dtype", "moe_dtype")

NUCLEUS_DTYPE = Literal["float32", "float16", "bfloat16"]


@dataclass
class Nucleus2MoEBaseConfig:
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


ConfigType = TypeVar("ConfigType", bound=Nucleus2MoEBaseConfig)


def config_to_dict(config: Nucleus2MoEBaseConfig) -> dict:
    """The config as a checkpoint stores it, with its dtypes as strings."""
    config_dict = dataclasses.asdict(config)
    for key in _DTYPE_FIELDS:
        if not isinstance(config_dict[key], str):
            config_dict[key] = _DTYPE_TO_STR[config_dict[key]]
    return config_dict


def config_from_dict(config_class: Type[ConfigType], config_dict: dict) -> ConfigType:
    """The config_class a checkpoint's config dict describes, with its dtypes as torch dtypes."""
    config_dict = dict(config_dict)
    for key in _DTYPE_FIELDS:
        if not isinstance(config_dict[key], torch.dtype):
            config_dict[key] = _STR_TO_DTYPE[config_dict[key]]
    return config_class(**config_dict)


def get_dtype(dtype):
    if isinstance(dtype, str):
        return _STR_TO_DTYPE[dtype]
    return dtype


class TransformerMoEBlock(nn.Module):
    def __init__(self, config: Nucleus2MoEBaseConfig, num_sim_params: int, drop_path_prob: float):
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


def _patch_embed(config: Nucleus2MoEBaseConfig, in_channels: int) -> nn.Module:
    if config.patching == "Linear":
        return LinearEmbed(patch_size=config.patch_size, in_channels=in_channels, embed_dim=config.embed_dim)
    return AdaptiveEmbed(in_channels=in_channels, out_channels=config.embed_dim, out_shape=(16, 16))


def patch_debed(config: Nucleus2MoEBaseConfig, out_channels: int) -> nn.Module:
    """An output head mapping the tokens of Nucleus2MoEBase.process back to
    out_channels on every cell, patched as config.patching sets."""
    if config.patching == "Linear":
        return LinearDebed(patch_size=config.patch_size, embed_dim=config.embed_dim, out_channels=out_channels)
    return AdaptiveDebed(in_channels=config.embed_dim, out_channels=out_channels, patch_shape=(16, 16))


class Nucleus2MoEBase(nn.Module):
    """The patch embedding and transformer MoE blocks every nucleus2 model shares.
    A derived model reads in_channels per cell, sets up its own output head, such
    as patch_debed for however many channels it predicts, and sets config_class,
    config_from_dict and field_names."""
    config_keys = NUCLEUS2_CONFIG_KEYS
    num_sim_params = num_config_columns(NUCLEUS2_CONFIG_KEYS)
    # config fields that set how the model is run rather than what it learned
    runtime_options: Tuple[str, ...] = ()

    def __init__(self, config: Nucleus2MoEBaseConfig, normalizer: NormalizerWrapper, in_channels: int):
        super().__init__()
        assert config.patching in ("Linear", "Adaptive")
        self.config = config
        self.normalizer = normalizer

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

        self.out_norm = CastRMSNorm(config.embed_dim)
        self.embed = _patch_embed(config, in_channels)

    def get_extra_state(self):
        return {"model_name": getattr(self, "_model_name", None), "config": config_to_dict(self.config)}

    def set_extra_state(self, state):
        self._model_name = state.get("model_name")
        self.config = restored_config(self.config, type(self).config_from_dict(state["config"]), self.runtime_options)

    @torch.no_grad()
    def predict_next(self, window: FlashXBatch, num_timesteps: int, target_timesteps: int) -> FlashXBatch:
        """The num_timesteps frames that follow a raw (unnormalized) window, raw, as the
        model's forward predicts them. The model was trained to predict
        target_timesteps frames after its input."""
        prediction = self(self.normalizer.normalize(window)).prediction
        return unnormalized_prediction(self.normalizer, window, next_frames(prediction, num_timesteps, target_timesteps))

    def process(self, cells: torch.Tensor, sim_params: torch.Tensor) -> Tuple[torch.Tensor, List[TopkMoEOutput]]:
        """cells: ``(B, T, H, W, in_channels)`` float32, sim_params: ``(B, num_sim_params)``.
        Returns the normalized patch tokens, ``(B, T, patches_y, patches_x, embed_dim)``
        in debed_dtype, for an output head, then the MoE output of every block."""
        assert cells.dtype == torch.float32, f"expected float32, got {cells.dtype}"
        assert sim_params.dtype == torch.float32, f"expected float32, got {sim_params.dtype}"

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

        with record_function("out_norm"):
            x = self.out_norm(x.to(self.debed_dtype) + embed.to(self.debed_dtype))
        return x, moe_outputs
