import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from nucleus.layers.adaptive_layernorm import AdaptiveLayerNorm
from nucleus.layers.space_time_attention import NeighborhoodAttention, SpaceTimeNeighborAttention, SpaceTimeAttention, SpaceTimeAxialAttention
from nucleus.layers.mlp import GeluMLP
from torch.profiler import record_function
from nucleus.layers.moe.topk_moe import TopkMoE, TopkMoEOutput, TopkRouterWithLoss, TopkRouterWithBias
from nucleus.layers.droppath import DropPath

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        drop_path_prob: float,
    ):
        super().__init__()
        
        self.attention_norm = nn.RMSNorm(embed_dim)
        self.mlp_norm = nn.RMSNorm(embed_dim)
        self.drop_path = DropPath(drop_path_prob)
        self.attention = SpaceTimeAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.mlp = GeluMLP(embed_dim)
    
    def _attention(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        with record_function("attention"):
            x = self.drop_path(self.attention(self.attention_norm(x), freqs)) + x
        return x
    
    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("mlp"):
            x = self.drop_path(self.mlp(self.mlp_norm(x))) + x
        return x    
    
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = self._attention(x, freqs)
        x = self._mlp(x)
        return x
    
class TransformerMoEBlock(TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        topk: int,
        drop_path_prob: float,
        num_fluid_params: int,
    ):
        super().__init__(embed_dim, num_heads, drop_path_prob)

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
        
        self.mlp = TopkMoE(
            num_experts=num_experts,
            hidden_dim=embed_dim,
            intermediate_dim=embed_dim * 4,
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
    
class TransformerNeighborBlock(TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        drop_path_prob: float,
    ):
        super().__init__(embed_dim, num_heads, drop_path_prob)
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
        drop_path_prob: float,
        num_fluid_params: int,
    ):
        super().__init__(
            embed_dim, num_heads, num_experts, topk, drop_path_prob, num_fluid_params
        )
        self.attention = NeighborhoodAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
    
class TransformerSpatialNeighborBlock(TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        drop_path_prob: float,
    ):
        super().__init__(embed_dim, num_heads, drop_path_prob)
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
        drop_path_prob: float,
        num_fluid_params: int,
    ):
        super().__init__(
            embed_dim, num_heads, num_experts, topk, drop_path_prob, num_fluid_params
        )
        self.attention = SpaceTimeNeighborAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
    
class TransformerAxialBlock(TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        drop_path_prob: float,
    ):
        super().__init__(embed_dim, num_heads, drop_path_prob)
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
        drop_path_prob: float,
        num_fluid_params: int,
    ):
        super().__init__(
            embed_dim, num_heads, num_experts, topk, drop_path_prob, num_fluid_params
        )
        self.attention = SpaceTimeAxialAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )