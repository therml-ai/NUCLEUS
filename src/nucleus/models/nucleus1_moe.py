import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import numpy as np
from einops import rearrange
from torch.profiler import record_function

from nucleus.layers import (
    HMLPEmbed, 
    HMLPDebed,
    FiLMMLP
)
from nucleus.layers import (
    Nucleus1TransformerMoEBlock, 
    Nucleus1TransformerAxialMoEBlock, 
    Nucleus1TransformerNeighborMoEBlock
)
from nucleus.utils.sdf_reinit import sdf_reinit_sussman

from nucleus.data.batching import CollatedBatch
from ._api import register_model

__all__ = ["Nucleus1ViTMoE", "Nucleus1AxialMoE", "Nucleus1NeighborMoE"]

class Nucleus1MoEBase(nn.Module):
    expected_fluid_params = [
        "inv_reynolds",
        "cpgas",
        "mugas",
        "rhogas",
        "thcogas",
        "stefan",
        "prandtl",
        "gravy",
        "bulk_temp",
        "sat_temp"
    ]
    expected_heater_params = [
        "wallTemp",
        "nucWaitTime",
        "rcdAngle",
        "advAngle",
        "velContact",
        "xMin",
        "xMax"
    ]
    expected_global_params = [
        "gravy"
    ]
    expected_fields = [
        "dfun", # sdf
        "temperature"
        "velx",
        "vely"
    ]
    num_sim_params = len(expected_fluid_params) + len(expected_heater_params) + len(expected_global_params)
    layout = "t c h w"
    
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_experts: int,
        topk: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )
        
        self.film_embed = FiLMMLP(self.num_sim_params, embed_dim)
        
        self.blocks = nn.ModuleList([
            Nucleus1TransformerMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                mlp_ratio=mlp_ratio,
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
        return self.step(batch.input, batch.sim_params_tensor)
        
    def step(self, input: torch.Tensor, sim_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
            sim_params: (B, num_sim_params)
        """
        x = input
        B, T, _, _, _ = x.shape
        
        input = x.clone()
        assert input.dtype == torch.float32
        assert sim_params.dtype == torch.float32

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

        # Apply FiLM conditioning on the embeddings
        with record_function("film_embed"):
            x = self.film_embed(x, sim_params)

        # Attention blocks, tracking the MoE output for the routing losses
        moe_outputs = []
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
               x, moe_output = blk(x)
               moe_outputs.append(moe_output)

        x = rearrange(x, "b t h w c -> b t c h w").contiguous()
        
        # Skip connection from patch embeddings
        x = x + embed
       
        # Decode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        x = nn.functional.gelu(x)
        
        # convert to float32 for high-precision output projection
        x = x.to(torch.float32)
        
        # project to output fields
        sdf = self.sdf_proj(x)
        temp = self.temp_proj(x)
        vel = self.vel_proj(x)
        sdf = rearrange(sdf, "(b t) c h w -> b t c h w", b=B, t=T)
        temp = rearrange(temp, "(b t) c h w -> b t c h w", b=B, t=T)
        vel = rearrange(vel, "(b t) c h w -> b t c h w", b=B, t=T)
        x = torch.cat((sdf, temp, vel), dim=2)
        
        # Skip connection from the input
        x = x + input
        
        return x, moe_outputs
    
    def forward_trajectory(
        self, 
        initial_state: torch.Tensor, 
        sim_params: torch.Tensor,
        dx: float,
        input_time_window_size: int,
        output_time_window_size: int,
        trajectory_steps: int,
        use_sdf_reinit: bool = False,
        return_moe_outputs: bool = False
    ):
        assert initial_state.dim() == 5, "initial state must be [B, T, C, H, W]"
        assert sim_params.dim() == 2, "fluid params must be [B, num_params]"
        assert initial_state.shape[0] == sim_params.shape[0]
        assert input_time_window_size == initial_state.shape[1]

        trajectory = initial_state.clone()
        trajectory_moe_outputs = [] if return_moe_outputs else None

        for _ in range(input_time_window_size, trajectory_steps, output_time_window_size):
            pred, moe_outputs = self.step(trajectory[:, -input_time_window_size:], sim_params)
            output_time_window = pred[:, -output_time_window_size:]
            
            if use_sdf_reinit:
                output_time_window[:, :, 0] = sdf_reinit_sussman(output_time_window[:, :, 0], dx=dx, n_iter=5)

            trajectory = torch.cat((trajectory, output_time_window), dim=1)
            if return_moe_outputs:
                trajectory_moe_outputs.append(moe_outputs)

        if return_moe_outputs:
            return trajectory, trajectory_moe_outputs
        return trajectory

    
@register_model("nucleus1_vit_moe")
class Nucleus1ViTMoE(Nucleus1MoEBase):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_experts: int,
        topk: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_experts=num_experts,
            topk=topk,
            mlp_ratio=mlp_ratio,
        )

@register_model("nucleus1_axial_moe")
class Nucleus1AxialMoE(Nucleus1MoEBase):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_experts: int,
        topk: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_experts=num_experts,
            topk=topk,
            mlp_ratio=mlp_ratio,
        )
        self.blocks = nn.ModuleList([
            Nucleus1TransformerAxialMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(processor_blocks)
        ])

@register_model("nucleus1_moe")
class Nucleus1NeighborMoE(Nucleus1MoEBase):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_experts: int,
        topk: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_experts=num_experts,
            topk=topk,
            mlp_ratio=mlp_ratio,
        )
        self.blocks = nn.ModuleList([
            Nucleus1TransformerNeighborMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(processor_blocks)
        ])
