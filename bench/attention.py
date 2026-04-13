import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import natten
natten.use_kv_parallelism_in_fused_na(mode=False)
natten.set_memory_usage_preference(pref='unrestricted')

from triton.testing import do_bench_cudagraph as do_bench

def spda(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v
        )

NATTEN_CONFIG = {
    "backend": "cutlass-fna",
    "q_tile_shape": (4, 16),
    "kv_tile_shape": (4, 16),
    "backward_q_tile_shape": (4, 16),
    "backward_kv_tile_shape": (4, 16),
}

@torch.compile
def neighborhood(q, k, v):
    return natten.na2d(
        q,
        k,
        v,
        kernel_size=3,
        stride=1,
        dilation=1,
        **NATTEN_CONFIG
    )
    
def train_step(module, q, k, v):
    out = module(q, k, v)
    loss = out.sum()
    loss.backward()
    return loss

batch_size = 32
time = 8
height = 64
width = 64
head_dim = 128
num_heads = 4
DEVICE = "cuda"

WARMUP = 100
REP = 200
quantiles = [0.25, 0.5, 0.75]

q = torch.randn(batch_size * time, num_heads, height * width, head_dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=False)
k = torch.randn(batch_size * time, num_heads, height * width, head_dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=False)
v = torch.randn(batch_size * time, num_heads, height * width, head_dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=False)

with torch.inference_mode():
    min_ms, ms, max_ms = do_bench(lambda: spda(q, k, v), quantiles=quantiles, rep=REP)
    print(f"Inference SPDA: {ms:.2f} ± {max_ms - min_ms:.2f} ms")

q = torch.randn(batch_size * time, height, width, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=False)
k = torch.randn(batch_size * time, height, width, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=False)
v = torch.randn(batch_size * time, height, width, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=False)


with torch.inference_mode():
    min_ms, ms, max_ms = do_bench(lambda: neighborhood(q, k, v), quantiles=quantiles, rep=REP)
    print(f"Inference NATTEN: {ms:.2f} ± {max_ms - min_ms:.2f} ms")