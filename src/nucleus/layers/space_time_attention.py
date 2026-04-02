import torch
import torch.nn as nn

from nucleus.layers.attention import (
    NeighborhoodAttention,
    TemporalAttention,
    SpatialNeighborhoodAttention, 
    SpatialAttention, 
    SpatialAxialAttention,
)

class SpaceTimeAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__()
        
        self.temporal = TemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        
        self.attention = SpatialAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.attention(x)
        return x

class SpaceTimeNeighborAttention(SpaceTimeAttention):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__(embed_dim, num_heads)
        
        self.attention = SpatialNeighborhoodAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.attention(x)
        return x
    
class SpaceTimeAxialAttention(SpaceTimeAttention):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__(embed_dim, num_heads)
        
        self.attention = SpatialAxialAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.attention(x)
        return x