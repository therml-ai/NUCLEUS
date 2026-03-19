import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import numpy as np
from einops import rearrange
from torch.profiler import record_function

from bubbleformer.layers import (
    HMLPEmbed, 
    HMLPDebed,
    LinearEmbed,
    LinearDebed,
    FiLMMLP,
    TransformerMoEBlock,
    TransformerAxialMoEBlock,
    TransformerNeighborMoEBlock
)
from bubbleformer.data.batching import CollatedBatch
from ._api import register_model

__all__ = ["ViTMoE", "AxialMoE", "NeighborMoE"]

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
        load_balance_loss_weight: float,
    ):
        super().__init__()
        self.embed = LinearEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )
        
        self.film_embed = FiLMMLP(num_fluid_params, embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                load_balance_loss_weight=load_balance_loss_weight,
            )
            for _ in range(processor_blocks)
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
        fluid_params = batch.fluid_params_tensor(x.device)
        B, T, _, _, _ = x.shape
        
        input = x
        assert input.dtype == torch.float32
        assert fluid_params.dtype == torch.float32
        
        # Encode
        with record_function("encode"):
            x = self.embed(x)
        embed = x

        # Apply FiLM conditioning on the embeddings
        with record_function("film_embed"):
            x = self.film_embed(x, fluid_params)
        fluid_embed = x

        # Attention blocks, tracking the MoE output for the routing losses
        moe_outputs = []
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
               x, moe_output = blk(x)
               moe_outputs.append(moe_output)
        
        # Skip connection from patch and fluid embeddings
        x = x + embed + fluid_embed        
        x = self.out_norm(x)
        x = self.debed(x)
        
        # Skip connection from the input
        x = x + input
        
        return x, moe_outputs
    
@register_model("vit_moe")
class ViTMoE(MoEBase):
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
        load_balance_loss_weight: float,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_fluid_params=num_fluid_params,
            num_experts=num_experts,
            topk=topk,
            load_balance_loss_weight=load_balance_loss_weight,
        )

@register_model("axial_moe")
class AxialMoE(MoEBase):
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
        load_balance_loss_weight: float,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_fluid_params=num_fluid_params,
            num_experts=num_experts,
            topk=topk,
            load_balance_loss_weight=load_balance_loss_weight,
        )
        self.blocks = nn.ModuleList([
            TransformerAxialMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                load_balance_loss_weight=load_balance_loss_weight,
            )
            for _ in range(processor_blocks)
        ])

@register_model("neighbor_moe")
class NeighborMoE(MoEBase):
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
        load_balance_loss_weight: float,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_fluid_params=num_fluid_params,
            num_experts=num_experts,
            topk=topk,
            load_balance_loss_weight=load_balance_loss_weight,
        )
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