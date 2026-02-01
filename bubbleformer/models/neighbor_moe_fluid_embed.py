import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import numpy as np
from einops import rearrange
from torch.profiler import record_function

from bubbleformer.data.batching import CollatedBatch
from bubbleformer.layers import (
    HMLPEmbed, 
    HMLPDebed,
    FiLMMLP,
    TransformerNeighborMoEBlock
)
from bubbleformer.layers.fluid_param_embedding import FluidParamEmbedding
from ._api import register_model

__all__ = ["NeighborMoEFluidEmbed"]

@register_model("neighbor_moe_fluid_embed")
class NeighborMoEFluidEmbed(nn.Module):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        time_window: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__()
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )
        
        self.fluid_param_embed = FluidParamEmbedding(embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerNeighborMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                load_balance_loss_weight=load_balance_loss_weight,
            )
            for _ in range(processor_blocks)
        ])

        self.debed = HMLPDebed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=embed_dim
        )
        
        self.sdf_proj = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1, dtype=torch.float32)
        self.temp_proj = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1, dtype=torch.float32)
        self.vel_proj = nn.Conv2d(embed_dim, 2, kernel_size=3, padding=1, dtype=torch.float32)
        
    def forward(self, batch: CollatedBatch) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        fluid_params: (B, num_fluid_params)
        """
        x = batch.input
        B, T, _, _, _ = x.shape
                
        input = x.clone()
        assert input.dtype == torch.float32
               
        # embed the fluid params as a (B, 1, 1, 1, embed_dim) tensor that can be added to each channel.
        fluid_param_embedding = self.fluid_param_embed(batch.fluid_params_dict)[:, None, None, None, :]

        # Encode
        with record_function("encode"):
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.embed(x)
            x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        embed = x.clone()

        # Permute to better order for attention (B, T, H, W, C)
        # TODO: IDK if input should be in this format for the embedding or not...
        # I think conv's do support NHWC layout.
        x = rearrange(x, "b t c h w -> b t h w c").contiguous()

        x += fluid_param_embedding

        # Attention blocks, tracking the MoE output for the routing losses
        moe_outputs = []
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
               x, moe_output = blk(x)
               x += fluid_param_embedding
               moe_outputs.append(moe_output)
        
        x = rearrange(x, "b t h w c -> b t c h w").contiguous()
        
        # Skip connection from patch embeddings
        x = x + embed
       
        # Decode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        x = nn.functional.gelu(x)
        
        # project to output fields
        sdf = self.sdf_proj(x)
        temp = self.temp_proj(x)
        vel = self.vel_proj(x)
        sdf = rearrange(sdf, "(b t) c h w -> b t c h w", b=B, t=T)
        temp = rearrange(temp, "(b t) c h w -> b t c h w", b=B, t=T)
        vel = rearrange(vel, "(b t) c h w -> b t c h w", b=B, t=T)
        x = torch.cat((sdf, temp, vel), dim=2)
        
        # Skip connection from the last timestep of the original input
        x = x + input[:, -1].unsqueeze(1).expand(-1, T, -1, -1, -1)
        
        return x, moe_outputs