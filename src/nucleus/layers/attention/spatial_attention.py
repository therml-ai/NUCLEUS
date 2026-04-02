import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import rearrange
import einops
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
import math
import time

@torch.compile(fullgraph=True)
class SpatialAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

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
        
        # rotary embedding expects seq-last [batch, heads, seq1, seq2, dim] layout
        heads = einops.rearrange(self.input_head(x.to(self.work_dtype)), 
                                 "b t h w (heads head_dim) -> (b t) heads h w head_dim", 
                                 heads=self.num_heads).contiguous()
        q, k, v = heads.tensor_split(3, dim=-1)
        q = self.qnorm(q)
        k = self.knorm(k)
        freqs = self.rotary_emb.get_axial_freqs(h, w)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        
        # SPDA expects sequence to be flattened [batch, heads, seq1 * seq2, dim] layout
        q, k, v = map(
            lambda qkv: rearrange(qkv, "bt heads h w head_dim -> bt heads (h w) head_dim").contiguous(), [q, k, v]
        )
        
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v
            )
             
        output = einops.rearrange(output,
                                  "(b t) heads (h w) head_dim -> b t h w (heads head_dim)", 
                                  b=b, t=t, h=h, w=w, heads=self.num_heads).contiguous()
        output = self.output_head(output).to(torch.float32)
        return output