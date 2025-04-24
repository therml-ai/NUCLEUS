import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.layers import DropPath

from bubbleformer.layers import AttentionBlock, HMLPEmbed, HMLPDebed, \
                                GeluMLP, RelativePositionBias, ContinuousPositionBias1D
from ._api import register_model

class ViTBlock(nn.Module):
    """
    Standard spatial attention block that applies attention over all spatial tokens
    
    Args:
        embed_dim (int): Number of features in the input tensor
        num_heads (int): Number of attention heads
        drop_path (float): Drop path rate
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        bias_type: str = "rel",
    ):
        super().__init__()

        # Use regular attention for spatial tokens but with a custom implementation
        # that doesn't use InstanceNorm2d which requires spatial dimensions > 1
        self.temporal = AttentionBlock(embed_dim, num_heads, drop_path=drop_path)

        # Spatial attention components - similar to AttentionBlock but for 1D spatial tokens
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.qnorm = nn.LayerNorm(embed_dim // num_heads)
        self.knorm = nn.LayerNorm(embed_dim // num_heads)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)
        self.gamma_att = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.gamma_mlp = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        if bias_type == "none":
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == "continuous":
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = GeluMLP(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            x (torch.Tensor): Input tensor of shape (B, T, Emb, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, Emb, H, W)
        """
        _, t, _, h, w = x.shape

        # First do temporal attention
        x = self.temporal(x)    # (B, T, emb, H, W)

        # Now do spatial attention by flattening H and W dimensions
        x = rearrange(x, "b t emb h w -> (b t) (h w) emb")  # BT, emb, HW

        # Store original input for skip connection
        inp = x.clone()
        _, n, _ = inp.shape

        # Apply normalization and project to Q, K, V
        x = self.norm1(x)  # BT, HW, emb
        x = self.qkv_proj(x)  # BT, HW, 3*emb
        x = rearrange(x, "b n (split he emb) -> split b he n emb", split=3, he=self.num_heads)
        q, k, v = x[0], x[1], x[2]  # Each: BT, He, HW, emb
        q, k = self.qnorm(q), self.knorm(k)

        rel_pos_bias = self.rel_pos_bias(n, n)

        # Apply scaled dot-product attention
        if rel_pos_bias is not None:
            # pylint: disable=not-callable
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=rel_pos_bias
            )
        else:
            # pylint: disable=not-callable
            x = F.scaled_dot_product_attention(
                query=q.contiguous(),
                key=k.contiguous(),
                value=v.contiguous(),
            )  # BT, He, HW, emb

        # Combine heads and project
        x = rearrange(x, "b he n emb -> b n (he emb)")  # BT, HW, emb
        x = self.norm2(x)
        x = self.output_proj(x)  # BT, HW, emb

        # Skip connection and scaling
        x = self.drop_path(x * self.gamma_att[None, None, :]) + inp

        # Apply MLP
        inp = x.clone()
        x = self.mlp(x)
        x = self.mlp_norm(x)
        x = self.drop_path(x * self.gamma_mlp[None, None, :]) + inp


        # Reshape back to original format
        x = rearrange(x, "(b t) (h w) emb -> b t emb h w", t=t, h=h, w=w) # BT, emb, HW

        return x    # (B, T, C, H, W)

@register_model("vit")
class ViT(nn.Module):
    """
    Vision Transformer model that applies standard spatial attention over flattened
    spatial tokens and temporal attention over time dimension.
    
    Args:
        fields (int): Number of fields
        time_window (int): Number of time steps
        patch_size (int): Size of the square patch
        embed_dim (int): Dimension of the embedding
        num_heads (int): Number of attention heads
        processor_blocks (int): Number of processor blocks
        drop_path (float): Dropout rate
    """
    def __init__(
        self,
        fields: int = 3,
        time_window: int = 12,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        processor_blocks: int = 12,
        drop_path: float = 0.2,
    ):
        super().__init__()
        self.drop_path = drop_path
        self.dp = np.linspace(0, drop_path, processor_blocks)
        # Hierarchical Patch Embedding
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=fields,
            embed_dim=embed_dim,
        )
        # Standard vision transformer blocks with spatial and temporal attention
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    drop_path=self.dp[i],
                    bias_type="rel"
                )
                for i in range(processor_blocks)
            ]
        )
        # Patch Debedding
        self.debed = HMLPDebed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=fields
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C, H, W)
        """
        _, t, _, _, _ = x.shape

        # Encode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)

        # Process
        for blk in self.blocks:
            x = blk(x)

        # Decode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)

        return x  # Temporal bundling (B, T, C, H, W)
