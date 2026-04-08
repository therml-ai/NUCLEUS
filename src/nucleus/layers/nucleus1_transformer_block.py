import torch
import torch.nn as nn
from nucleus.layers.nucleus1_space_time_attention import (
    Nucleus1SpaceTimeNeighborAttention, 
    Nucleus1SpaceTimeAttention, 
    Nucleus1SpaceTimeAxialAttention,
)
from nucleus.layers.mlp import GeluMLP
from nucleus.layers.moe.nucleus1_topk_moe import TopkMoE, TopkMoEOutput
from torch.profiler import record_function

class Nucleus1TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__()
        
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)
        
        self.attention = Nucleus1SpaceTimeAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        
        self.mlp = GeluMLP(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("transformer_block"):
            # Attention with a skip connection
            inp = x.clone()
            with record_function("space_time_attention"):
                x = self.attention(x)
            with record_function("pre_norm"):          
                x = self.pre_norm(x) + inp

            # MLP with a skip connection
            intermediate = x.clone()
            with record_function("mlp"):
                x = self.mlp(x)
            with record_function("post_norm"):
                x = self.post_norm(x) + intermediate
        return x

class Nucleus1TransformerMoEBlock(Nucleus1TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__(embed_dim, num_heads)
    
        self.mlp = TopkMoE(
            num_experts=num_experts,
            hidden_dim=embed_dim,
            intermediate_dim=embed_dim * 4,
            topk=topk,
            load_balance_loss_weight=load_balance_loss_weight,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("transformer_moe_block"):
            # Attention with a skip connection
            inp = x.clone()
            with record_function("space_time_attention"):
                x = self.attention(x)
            with record_function("pre_norm"):          
                x = self.pre_norm(x) + inp

            # MLP with a skip connection
            intermediate = x.clone()
            with record_function("mlp"):
                moe_output: TopkMoEOutput = self.mlp(x)
                x = moe_output.out
            with record_function("post_norm"):
                x = self.post_norm(x) + intermediate

        return x, moe_output


class Nucleus1TransformerNeighborBlock(Nucleus1TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__(embed_dim, num_heads)
        self.attention = Nucleus1SpaceTimeNeighborAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.mlp = GeluMLP(embed_dim)
    
class Nucleus1TransformerNeighborMoEBlock(Nucleus1TransformerMoEBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__(embed_dim, num_heads, num_experts, topk, load_balance_loss_weight)
        self.attention = Nucleus1SpaceTimeNeighborAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
    
class Nucleus1TransformerAxialBlock(Nucleus1TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__(embed_dim, num_heads)
        self.attention = Nucleus1SpaceTimeAxialAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )    
    
class Nucleus1TransformerAxialMoEBlock(Nucleus1TransformerMoEBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__(embed_dim, num_heads, num_experts, topk, load_balance_loss_weight)
        self.attention = Nucleus1SpaceTimeAxialAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )