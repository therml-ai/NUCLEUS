import torch
import torch.nn as nn
from torch.profiler import record_function
from rotary_embedding_torch import RotaryEmbedding

from nucleus.layers.adaptive_layernorm import AdaptiveLayerNorm
from nucleus.layers.mlp import GeluMLP
from nucleus.layers.attention import NeighborhoodAttention
from nucleus.layers.moe.topk_moe import TopkMoE, TopkMoEOutput, TopkRouterWithBias
from nucleus.layers.droppath import DropPath
from nucleus.layers import (
    LinearEmbed,
    LinearDebed,
    TransformerMoEBlock,
)
from nucleus.data.batching import CollatedBatch
from ._api import register_model

__all__ = ["NeighborMoE"]
    
class TransformerMoEBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        topk: int,
        drop_path_prob: float,
        num_fluid_params: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__(embed_dim, num_heads, drop_path_prob, mlp_ratio=mlp_ratio)

        self.drop_path = DropPath(drop_path_prob)

        self.attention_norm = AdaptiveLayerNorm(embed_dim, num_fluid_params)
        self.mlp_norm = AdaptiveLayerNorm(embed_dim, num_fluid_params)

        self.router = TopkRouterWithBias(
            num_experts, 
            embed_dim, 
            topk, 
            bias_update_rate=0.001, # This was used in deepseek-v3
            softmax_first=False
        )
        
        self.attention = NeighborhoodAttention(embed_dim=embed_dim, num_heads=num_heads)
        
        self.mlp = TopkMoE(
            num_experts=num_experts,
            hidden_dim=embed_dim,
            intermediate_dim=int(embed_dim * mlp_ratio),
            topk=topk,
            router=self.router
        )

    def _attention(
        self, x: torch.Tensor, freqs: torch.Tensor, fluid_params: torch.Tensor
    ) -> torch.Tensor:
        with record_function("attention"):
            x = x + self.drop_path(self.attention(self.attention_norm(x, fluid_params), freqs))
        return x
    
    def _mlp(self, x: torch.Tensor, fluid_params: torch.Tensor) -> torch.Tensor:
        with record_function("moe"):
            moe_output: TopkMoEOutput = self.mlp(self.mlp_norm(x, fluid_params))
            x = x + self.drop_path(moe_output.out)
        return x, moe_output
        
    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        fluid_params: torch.Tensor,
    ) -> torch.Tensor:
        x = self._attention(x, freqs, fluid_params)
        x, moe_output = self._mlp(x, fluid_params)
        return x, moe_output

class MoEBase(nn.Module):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
        num_experts: int,
        topk: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed = LinearEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )
        
        # Every attention block reuses the same frequencies, so we only need to compute them once.
        self.rotary_emb = RotaryEmbedding(
            # NOTE: This must be smaller than the head dim. 
            dim=(embed_dim // num_heads) // 3,
            freqs_for="pixel",
            max_freq=256,
            # We want a [Batch, Seq1, Seq2, Seq3, Heads, Dim] layout
            seq_before_head_dim=True
        )
        
        self.drop_path_probs = torch.linspace(0.0, 0.1, processor_blocks)
        
        self.blocks = nn.ModuleList([
            TransformerMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                drop_path_prob=self.drop_path_probs[idx].item(),
                num_fluid_params=num_fluid_params,
                mlp_ratio=mlp_ratio,
            )
            for idx in range(processor_blocks)
        ])

        self.out_norm = nn.RMSNorm(embed_dim)
        self.debed = LinearDebed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=output_fields
        )        
        
    def forward(self, batch: CollatedBatch) -> torch.Tensor:
        """
        x: (B, T, H, W, C)
        fluid_params: (B, num_fluid_params)
        """
        x = batch.input
        input = x
        assert input.dtype == torch.float32
        
        with record_function("encode"):
            x = self.embed(x)
        embed = x
        
        # Get axial frequencies for rotary embedding.
        # We expand the dims so that it matches [B, T, H, W, heads, head_dim] used in the attention layers.
        # These are unlearned, so do with no_grad.
        with record_function("get_axial_freqs"):
            with torch.no_grad():
                _, embed_t, embed_h, embed_w, _ = embed.shape
                rotary_freqs = self.rotary_emb.get_axial_freqs(embed_t, embed_h, embed_w)[None, :, :, :, None, :]

        # Attention blocks, tracking the MoE output for the routing losses
        moe_outputs = []
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
               x, moe_output = blk(x, rotary_freqs, batch.fluid_params_tensor)
               moe_outputs.append(moe_output)
        
        # Skip connections from patch embeddings
        x = x + embed 

        with record_function("debed"):
            x = self.out_norm(x)
            x = self.debed(x)

        return x, moe_outputs

@register_model("neighbor_moe")
class NeighborMoE(MoEBase):
    pass

@register_model("nucleus2_moe")
class Nucleus2MoE(MoEBase):
    pass