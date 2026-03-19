import torch
import torch.nn as nn
from bubbleformer.layers.space_time_attention import NeighborhoodAttention, SpaceTimeNeighborAttention, SpaceTimeAttention, SpaceTimeAxialAttention
from bubbleformer.layers.mlp import GeluMLP
from torch.profiler import record_function
from bubbleformer.layers.moe.topk_moe import TopkMoE, TopkMoEOutput, TopkRouterWithLoss, TopkRouterWithBias

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__()
        
        self.attention_norm = nn.RMSNorm(embed_dim)
        self.mlp_norm = nn.RMSNorm(embed_dim)
        
        self.attention = SpaceTimeAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        
        self.mlp = GeluMLP(embed_dim)
        
    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("attention"):
            x = self.attention(self.attention_norm(x)) + x
        return x
    
    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("mlp"):
            x = self.mlp(self.mlp_norm(x)) + x
        return x    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._attention(x)
        x = self._mlp(x)
        return x
    
class TransformerMoEBlock(TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__(embed_dim, num_heads)

        self.router = TopkRouterWithBias(
            num_experts, 
            embed_dim, 
            topk, 
            bias_update_rate=0.001, # This was used in deepseek-v3
            softmax_first=False
        )
        
        self.mlp = TopkMoE(
            num_experts=num_experts,
            hidden_dim=embed_dim,
            intermediate_dim=embed_dim * 4,
            topk=topk,
            router=self.router
        )
        
    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("mlp"):
            moe_output: TopkMoEOutput = self.mlp(x)
            x = moe_output.out
        return x, moe_output
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._attention(x)
        x, moe_output = self._mlp(x)
        return x, moe_output
    
class TransformerNeighborBlock(TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__(embed_dim, num_heads)
        self.attention = NeighborhoodAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
    
class TransformerNeighborMoEBlock(TransformerMoEBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__(embed_dim, num_heads, num_experts, topk, load_balance_loss_weight)
        self.attention = NeighborhoodAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
    
class TransformerSpatialNeighborBlock(TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__(embed_dim, num_heads)
        self.attention = SpaceTimeNeighborAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.mlp = GeluMLP(embed_dim)
    
class TransformerSpatialNeighborMoEBlock(TransformerMoEBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__(embed_dim, num_heads, num_experts, topk, load_balance_loss_weight)
        self.attention = SpaceTimeNeighborAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
    
class TransformerAxialBlock(TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__(embed_dim, num_heads)
        self.attention = SpaceTimeAxialAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )    
    
class TransformerAxialMoEBlock(TransformerMoEBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__(embed_dim, num_heads, num_experts, topk, load_balance_loss_weight)
        self.attention = SpaceTimeAxialAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )