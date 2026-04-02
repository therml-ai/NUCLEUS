import torch
import torch.nn as nn


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, embed_dim: int, num_fluid_params: int, eps: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim, eps=eps, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.LayerNorm(num_fluid_params),
            nn.Linear(num_fluid_params, embed_dim * 2),
        )

    def forward(self, x: torch.Tensor, fluid_params: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected last dim {self.embed_dim}, got {x.shape[-1]}"
            )
        if fluid_params.shape[0] != x.shape[0]:
            raise ValueError(
                "Batch size of x and fluid_params must match "
                f"({x.shape[0]} vs {fluid_params.shape[0]})"
            )

        x = self.norm(x)
        gb = self.modulation(fluid_params)
        gamma, beta = gb.chunk(2, dim=-1)
        shape = [fluid_params.shape[0]] + [1] * (x.ndim - 2) + [self.embed_dim]
        gamma = gamma.view(shape)
        beta = beta.view(shape)
        return (1 + gamma) * x + beta
