import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import numpy as np
from einops import rearrange
from torch.profiler import record_function

from nucleus.layers import (
    HMLPEmbed, 
    HMLPDebed, 
    FiLMMLP,
    TransformerBlock,
    TransformerAxialBlock,
    TransformerNeighborBlock,
)
from nucleus.data.batching import CollatedBatch
from ._api import register_model

__all__ = ["ViT"]

class ViTBase(nn.Module):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
    ):
        super().__init__()
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )
        super().__init__()
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )

        self.film_embed = FiLMMLP(num_fluid_params, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
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
        fluid_params = batch.fluid_params_tensor(x.device)
        B, T, _, _, _ = x.shape
        
        input = x.clone()

        # Encode
        with record_function("encode"):
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.embed(x)
            x = rearrange(x, "(b t) c h w -> b t c h w", t=T)
            
        embed = x.clone()

        x = rearrange(x, "b t c h w -> b t h w c").contiguous()

        # Apply FiLM conditioning on the embeddings
        with record_function("film_embed"):
            x = self.film_embed(x, fluid_params)

        # Attention blocks
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
               x = blk(x)

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
        
        return x
        
@register_model("vit")
class ViT(ViTBase):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_fluid_params=num_fluid_params,
        )

@register_model("axial_vit")
class AxialViT(ViTBase):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_fluid_params=num_fluid_params,
        )
        self.blocks = nn.ModuleList([
            TransformerAxialBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
            for _ in range(processor_blocks)
        ])
    
@register_model("neighbor_vit")
class NeighborViT(ViTBase):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_fluid_params=num_fluid_params,
        )
        self.blocks = nn.ModuleList([
            TransformerNeighborBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
            for _ in range(processor_blocks)
        ])