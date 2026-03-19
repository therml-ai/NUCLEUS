import torch
import torch.nn as nn
from einops import rearrange
import einops
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

import natten

# NOTE: due to natten, this cannot be compiled with fullgraph=True
@torch.compile
class NeighborhoodAttention(nn.Module):
    r"""
    This is similar to natten's NaighborhoodAttention2D,
    but includes additional query and key normalization.
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kernel_size = kernel_size

        self.input_head = nn.Linear(embed_dim, 3 * embed_dim, dtype=torch.bfloat16, bias=False)
        self.output_head = nn.Linear(embed_dim, embed_dim, dtype=torch.bfloat16, bias=False)
        self.qnorm = nn.RMSNorm(self.head_dim, dtype=torch.bfloat16)
        self.knorm = nn.RMSNorm(self.head_dim, dtype=torch.bfloat16)
        
        # TODO: should each attention block use the same rotary embedding?
        self.rotary_emb = RotaryEmbedding(
            # NOTE: This must be smaller than the head dim. 
            dim=self.head_dim // 3,
            freqs_for="pixel",
            max_freq=256
        )
        
        self.work_dtype = torch.bfloat16

    def forward(self, x):
        b, t, h, w, c = x.shape
        input_dtype = x.dtype
        
        # rotary embedding expects seq-last [batch, heads, seq1, seq2, seq3, dim] layout
        heads = einops.rearrange(self.input_head(x.to(self.work_dtype)), 
                                 "b t h w (heads head_dim) -> b heads t h w head_dim", 
                                 heads=self.num_heads).contiguous()
        q, k, v = heads.tensor_split(3, dim=-1)
        q = self.qnorm(q)
        k = self.knorm(k)
        freqs = self.rotary_emb.get_axial_freqs(t, h, w)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        
        # natten expects head-last [batch, seq1, seq2, heads, dim] layout
        q, k, v = map(
            lambda qkv: rearrange(qkv, "b heads t h w head_dim -> b t h w heads head_dim").contiguous(), [q, k, v]
        )
        
        output = natten.na3d(
            q,
            k,
            v,
            kernel_size=(t, self.kernel_size, self.kernel_size),
            stride=1,
            dilation=1,
        )
        
        output = output.view(b, t, h, w, self.num_heads * self.head_dim)
        output = self.output_head(output).to(input_dtype)
        return output