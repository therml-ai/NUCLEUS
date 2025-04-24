import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.layers import DropPath
from bubbleformer.layers import GeluMLP, RelativePositionBias, ContinuousPositionBias1D


class AttentionBlock(nn.Module):
    """
    Attention Block
    Takes in tensors of shape (B, n, emb, H, W)
    where n is the token sequence, emb is the embedding dimension
    H and W are the spatial tokens
    and applies self-attention across time dimension
    Args:
        embed_dim (int): Number of features in the input tensor
        num_heads (int): Number of attention heads
        drop_path (float): Drop path rate
        layer_scale_init_value (float): Initial value for layer scale
        bias_type (str): Type of bias to use in the attention mechanism
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        drop_path: float = 0,
        layer_scale_init_value: float = 1e-6,
        bias_type: str = "rel",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.InstanceNorm2d(embed_dim, affine=True)
        self.norm2 = nn.InstanceNorm2d(embed_dim, affine=True)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0 else None
        )
        self.input_head = nn.Conv2d(embed_dim, 3 * embed_dim, 1)
        self.output_head = nn.Conv2d(embed_dim, embed_dim, 1)
        self.qnorm = nn.LayerNorm(embed_dim // num_heads)
        self.knorm = nn.LayerNorm(embed_dim // num_heads)
        if bias_type == "none":
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == "continuous":
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, emb, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, N, emb, H, W)
        """
        _, n, _, h, w = x.shape
        inp = x.clone()
        # Rearrange and prenorm
        x = rearrange(x, "b n emb h w -> (b n) emb h w")
        x = self.norm1(x)
        x = self.input_head(x)  # Q, K, V projections
        # Rearrange for attention
        x = rearrange(x, "(b n) (he emb) h w ->  (b h w) he n emb", n=n, he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)
        rel_pos_bias = self.rel_pos_bias(n, n)
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
                value=v.contiguous()
            )
        # Rearrange after attention
        x = rearrange(x, "(b h w) he n emb -> (b n) (he emb) h w", h=h, w=w)
        x = self.norm2(x)
        x = self.output_head(x)
        x = rearrange(x, "(b n) emb h w -> b n emb h w", n=n)
        output = self.drop_path(x * self.gamma[None, None, :, None, None]) + inp
        return output


class AxialAttentionBlock(nn.Module):
    """
    Axial Attention Block
    Args:
        embed_dim (int):Embedding dimension
        num_heads (int): Number of attention heads
        drop_path (float): Dropout rate
        layer_scale_init_value (float): Initial value for layer scale
        bias_type (str): Type of bias to use
    """
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        drop_path=0,
        layer_scale_init_value=1e-6,
        bias_type="rel",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.InstanceNorm2d(embed_dim, affine=True)
        self.norm2 = nn.InstanceNorm2d(embed_dim, affine=True)
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

        self.input_head = nn.Conv2d(embed_dim, 3 * embed_dim, 1)
        self.output_head = nn.Conv2d(embed_dim, embed_dim, 1)
        self.qnorm = nn.LayerNorm(embed_dim // num_heads)
        self.knorm = nn.LayerNorm(embed_dim // num_heads)
        if bias_type == "none":
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == "continuous":
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = GeluMLP(embed_dim)
        self.mlp_norm = nn.InstanceNorm2d(embed_dim, affine=True)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, emb, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, emb, H, W)
        """
        _, _, h, w = x.shape
        inp = x.clone()
        x = self.norm1(x)
        x = self.input_head(x)

        x = rearrange(x, "b (he emb) h w ->  b he h w emb", he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)

        # Do attention with current q, k, v matrices along each spatial axis then average results
        # X direction attention
        qx, kx, vx = map(
            lambda x: rearrange(x, "b he h w emb ->  (b h) he w emb"), [q, k, v]
        )
        rel_pos_bias_x = self.rel_pos_bias(w, w)
        if rel_pos_bias_x is not None:
            # pylint: disable=not-callable
            xx = F.scaled_dot_product_attention(
                query=qx,
                key=kx,
                value=vx,
                attn_mask=rel_pos_bias_x,
            )
        else:
            # pylint: disable=not-callable
            xx = F.scaled_dot_product_attention(
                query=qx.contiguous(),
                key=kx.contiguous(),
                value=vx.contiguous(),
            )
        xx = rearrange(xx, "(b h) he w emb -> b (he emb) h w", h=h)

        # Y direction attention
        qy, ky, vy = map(
            lambda x: rearrange(x, "b he h w emb ->  (b w) he h emb"), [q, k, v]
        )
        rel_pos_bias_y = self.rel_pos_bias(h, h)
        if rel_pos_bias_y is not None:
            # pylint: disable=not-callable
            xy = F.scaled_dot_product_attention(
                query=qy,
                key=ky,
                value=vy,
                attn_mask=rel_pos_bias_y,
            )
        else:
            # pylint: disable=not-callable
            xy = F.scaled_dot_product_attention(
                query=qy.contiguous(),
                key=ky.contiguous(),
                value=vy.contiguous(),
            )
        xy = rearrange(xy, "(b w) he h emb -> b (he emb) h w", w=w)

        # Combine
        x = (xx + xy) / 2
        x = self.norm2(x)
        x = self.output_head(x)
        x = self.drop_path(x * self.gamma_att[None, :, None, None]) + inp

        # MLP
        inp = x.clone()
        x = rearrange(x, "b emb h w -> b h w emb")
        x = self.mlp(x)
        x = rearrange(x, "b h w emb -> b emb h w")
        x = self.mlp_norm(x)
        out = inp + self.drop_path(self.gamma_mlp[None, :, None, None] * x)

        return out
