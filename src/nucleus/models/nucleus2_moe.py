import dataclasses
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.profiler import record_function
from rotary_embedding_torch import RotaryEmbedding
from typing import List, Literal

from nucleus.layers.adaptive_layernorm import AdaptiveLayerNorm
from nucleus.layers.attention import NeighborhoodAttention
from nucleus.layers.moe.topk_moe import TopkMoE, TopkMoEOutput, TopkRouterWithBias
from nucleus.layers.droppath import DropPath
from nucleus.layers.cast import CastRMSNorm
from nucleus.layers import (
    AdaptiveEmbed,
    AdaptiveDebed,
    LinearEmbed,
    LinearDebed
)
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper
from nucleus.models.flashx_io import (
    NUCLEUS2_CONFIG_KEYS,
    num_config_columns,
    next_frames,
    require_normalized,
    restored_config,
    unnormalized_prediction,
)

from ._api import register_model

__all__ = ["Nucleus2MoE", "Nucleus2MoEConfig", "Nucleus2MoEOutput"]


_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
_STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}


NUCLEUS_DTYPE = Literal["float32", "float16", "bfloat16"]


@dataclass
class Nucleus2MoEConfig:
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


def _config_to_dict(config: Nucleus2MoEConfig) -> dict:
    d = dataclasses.asdict(config)
    dtype_fields = {"embed_dtype", "debed_dtype", "activation_dtype", "attention_dtype", "moe_dtype"}
    for key in dtype_fields:
        if not isinstance(d[key], str):
            d[key] = _DTYPE_TO_STR[d[key]]
    return d


def _config_from_dict(d: dict) -> Nucleus2MoEConfig:
    d = dict(d)
    dtype_fields = {"embed_dtype", "debed_dtype", "activation_dtype", "attention_dtype", "moe_dtype"}
    for key in dtype_fields:
        if not isinstance(d[key], torch.dtype):
            d[key] = _STR_TO_DTYPE[d[key]]
    return Nucleus2MoEConfig(**d)

def get_dtype(dtype):
    if isinstance(dtype, str):
        return _STR_TO_DTYPE[dtype]
    return dtype


class TransformerMoEBlock(nn.Module):
    def __init__(self, config: Nucleus2MoEConfig, num_sim_params: int, drop_path_prob: float):
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


@dataclass
class Nucleus2MoEOutput:
    prediction: FlashXBatch  # normalized; every frame of the output window
    moe_outputs: List[TopkMoEOutput]


@register_model("nucleus2_moe")
class Nucleus2MoE(nn.Module):
    config_class = Nucleus2MoEConfig
    config_from_dict = staticmethod(_config_from_dict)
    config_keys = NUCLEUS2_CONFIG_KEYS
    field_names = ("dfun", "temperature", "velx", "vely")
    # normalized with one shared scale and no offset, so a flip's sign change holds
    vector_fields = (("velx", "vely"),)
    # config fields that set how the model is run rather than what it learned
    runtime_options = ()
    num_sim_params = num_config_columns(NUCLEUS2_CONFIG_KEYS)

    def __init__(self, config: Nucleus2MoEConfig, normalizer: NormalizerWrapper):
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        n_fields = len(self.field_names)

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
        
        assert config.patching in ("Linear", "Adaptive")
        if config.patching == "Linear":    
            self.embed = LinearEmbed(
                patch_size=config.patch_size,
                in_channels=n_fields,
                embed_dim=config.embed_dim
            )
            self.debed = LinearDebed(
                patch_size=config.patch_size,
                embed_dim=config.embed_dim,
                out_channels=n_fields
            )
        else:          
            self.embed = AdaptiveEmbed(
                in_channels=n_fields,
                out_channels=config.embed_dim,
                out_shape=(16, 16)
            )
            self.debed = AdaptiveDebed(
                in_channels=config.embed_dim,
                out_channels=n_fields,
                patch_shape=(16, 16)
            )

    def get_extra_state(self):
        return {"model_name": getattr(self, "_model_name", None), "config": _config_to_dict(self.config)}

    def set_extra_state(self, state):
        self._model_name = state.get("model_name")
        self.config = restored_config(self.config, _config_from_dict(state["config"]), self.runtime_options)

    def forward(self, batch: FlashXBatch) -> Nucleus2MoEOutput:
        """batch: normalized, holding field_names and a config tensor of config_keys."""
        require_normalized(batch)
        output, moe_outputs = self.step(batch.stacked_fields(self.field_names), batch.config_tensor())
        prediction = batch.with_fields(dict(zip(self.field_names, output.unbind(-1))))
        return Nucleus2MoEOutput(prediction, moe_outputs)

    @torch.no_grad()
    def predict_next(self, window: FlashXBatch, num_timesteps: int, target_timesteps: int) -> FlashXBatch:
        """The num_timesteps frames that follow a raw (unnormalized) window, raw. The
        model was trained to predict target_timesteps frames after its input."""
        prediction = self(self.normalizer.normalize(window)).prediction
        return unnormalized_prediction(self.normalizer, window, next_frames(prediction, num_timesteps, target_timesteps))

    def step(self, input: torch.Tensor, sim_params: torch.Tensor):
        """
        input: (B, T, H, W, C)
        sim_params: (B, num_sim_params)
        """
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
            x = self.debed(x, target_shape=(h, w))

        return x.to(torch.float32), moe_outputs
