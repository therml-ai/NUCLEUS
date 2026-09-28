import torch
import torch.nn as nn

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, embed_dim: int, num_sim_params: int, dtype: torch.dtype, eps: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim, eps=eps, elementwise_affine=False)
        
        # modulation is applied in dtype
        self.dtype = dtype
        
        # Modulation is computed as float32
        self.modulation = nn.Sequential(
            nn.LayerNorm(num_sim_params),
            nn.Linear(num_sim_params, embed_dim * 2),
        )
        # Start as identity: gamma = beta = 0 so (1 + gamma) * x + beta == x.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, sim_params: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected last dim {self.embed_dim}, got {x.shape[-1]}"
            )
        if sim_params.shape[0] != x.shape[0]:
            raise ValueError(
                "Batch size of x and sim_params must match "
                f"({x.shape[0]} vs {sim_params.shape[0]})"
            )
        assert sim_params.dtype == torch.float32

        x = self.norm(x)
        gb = self.modulation(sim_params).to(self.dtype)
        gamma, beta = gb.chunk(2, dim=-1)
        shape = [sim_params.shape[0]] + [1] * (x.ndim - 2) + [self.embed_dim]
        gamma = gamma.view(shape)
        beta = beta.view(shape)
        return (1 + gamma) * x + beta
