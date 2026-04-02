import xml
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nucleus.layers import GeluMLP, RelativePositionBias, ContinuousPositionBias1D


class BubbleformerAttentionBlock(nn.Module):
    """
    Attention Block with Optional Scaling for High-Frequency Components
    Takes in tensors of shape (B, n, emb, H, W)
    where n is the token sequence, emb is the embedding dimension
    H and W are the spatial tokens
    and applies self-attention across time dimension
    Args:
        embed_dim (int): Number of features in the input tensor
        num_heads (int): Number of attention heads
        layer_scale_init_value (float): Initial value for layer scale
        bias_type (str): Type of bias to use in the attention mechanism
        attn_scale (bool): Whether to apply attention scaling
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        layer_scale_init_value: float = 1e-6,
        bias_type: str = "rel",
        attn_scale: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_scale = attn_scale

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

        # Initialize the attention scaling factor
        if attn_scale:
            self.attn_scale_factor = nn.Parameter(
                torch.ones((1, num_heads, 1, 1)), requires_grad=True
            )

        if bias_type == "none":
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == "continuous":
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)

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
        q, k, v = x.tensor_split(3, dim=-1)      # [bhw, he, n, emb//he]
        q, k = self.qnorm(q), self.knorm(k)
        rel_pos_bias = self.rel_pos_bias(n, n)  # [(1, num_heads, n, n)]

        if self.attn_scale:
            head_dim = self.embed_dim // self.num_heads
            scaling = head_dim ** -0.5
            # Attention with high frequency scaling
            if rel_pos_bias is not None:
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling + rel_pos_bias
            else:
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
            attn = F.softmax(attn_scores, dim=-1)

            attn_low = torch.ones(attn.shape[-2:], device=attn.device) / n    # [n, n]
            attn_low = attn_low[None, None, :, :]                             # [1, 1, n, n]
            attn_high = attn - attn_low
            attn_high = attn_high * self.attn_scale_factor
            attn = attn_low + attn_high                       # [bhw, num_heads, n, emb//he]

            x = torch.matmul(attn, v)
        else:
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
        return x


class BubbleformerAxialAttentionBlock(nn.Module):
    """
    Axial Attention Block
    Args:
        embed_dim (int):Embedding dimension
        num_heads (int): Number of attention heads
        layer_scale_init_value (float): Initial value for layer scale
        bias_type (str): Type of bias to use
    """
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        layer_scale_init_value=1e-6,
        bias_type="rel",
        attn_scale=True,
        feat_scale=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_scale = attn_scale
        self.feat_scale = feat_scale

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

        # Initialize the attention scaling factor and feature scaling factors
        if attn_scale:
            self.attn_scale_factor_x = nn.Parameter(
                torch.ones((1, num_heads, 1, 1)),
                requires_grad=True
            )
            self.attn_scale_factor_y = nn.Parameter(
                torch.ones((1, num_heads, 1, 1)),
                requires_grad=True
            )
        if feat_scale:
            self.low_freq_scalar = nn.Parameter(torch.zeros(embed_dim), requires_grad=True)
            self.high_freq_scalar = nn.Parameter(torch.zeros(embed_dim), requires_grad=True)

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

        if self.attn_scale:
            head_dim = self.embed_dim // self.num_heads
            scaling = head_dim ** -0.5
            # Attention with high frequency scaling
            if rel_pos_bias_x is not None:
                attn_scores = torch.matmul(qx, kx.transpose(-2, -1)) * scaling + rel_pos_bias_x
            else:
                attn_scores = torch.matmul(qx, kx.transpose(-2, -1)) * scaling
            attn = F.softmax(attn_scores, dim=-1)

            attn_low = torch.ones(attn.shape[-2:], device=attn.device) / w
            attn_low = attn_low[None, None, :, :]
            attn_high = attn - attn_low
            attn_high = attn_high * self.attn_scale_factor_x
            xx = attn_low + attn_high
            xx = torch.matmul(xx, vx)
        else:
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
        if self.attn_scale:
            head_dim = self.embed_dim // self.num_heads
            scaling = head_dim ** -0.5
            # Attention with high frequency scaling
            if rel_pos_bias_y is not None:
                attn_scores = torch.matmul(qy, ky.transpose(-2, -1)) * scaling + rel_pos_bias_y
            else:
                attn_scores = torch.matmul(qy, ky.transpose(-2, -1)) * scaling
            attn = F.softmax(attn_scores, dim=-1)

            attn_low = torch.ones(attn.shape[-2:], device=attn.device) / h
            attn_low = attn_low[None, None, :, :]
            attn_high = attn - attn_low
            attn_high = attn_high * self.attn_scale_factor_y
            xy = attn_low + attn_high
            xy = torch.matmul(xy, vy)
        else:
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

        # Feature scaling
        if self.feat_scale:
            x_low = torch.mean(x, dim=[2, 3], keepdim=True)   # [b, emb, 1, 1]
            x_high = x - x_low                                # [b, emb, h, w]
            x_low = x_low * self.low_freq_scalar[None, :, None, None]
            x_high = x_high * self.high_freq_scalar[None, :, None, None]
            x = x + x_low + x_high

        # MLP
        inp = x.clone()
        x = rearrange(x, "b emb h w -> b h w emb")
        x = self.mlp(x)
        x = rearrange(x, "b h w emb -> b emb h w")
        x = self.mlp_norm(x) + inp

        return x