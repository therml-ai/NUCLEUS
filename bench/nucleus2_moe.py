import argparse
import torch
from triton.testing import do_bench

from nucleus.models import Nucleus2MoE, Nucleus2MoEConfig
from nucleus.data.batching import CollatedBatch

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--patch_size", type=int, default=32)
parser.add_argument("--embed_dim", type=int, default=768)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--processor_blocks", type=int, default=12)
parser.add_argument("--moe_intermediate_dim", type=int, default=1024)
parser.add_argument("--backward", action="store_true", default=False)
parser.add_argument("-t", type=int, default=8)
parser.add_argument("--height", type=int, default=64)
parser.add_argument("--width", type=int, default=64)
args = parser.parse_args()

config = Nucleus2MoEConfig(
    patch_size=args.patch_size,
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    processor_blocks=args.processor_blocks,
    num_experts=6,
    topk=2,
    moe_intermediate_dim=args.moe_intermediate_dim,
    embed_dtype=torch.bfloat16,
    debed_dtype=torch.bfloat16,
    activation_dtype=torch.torch.bfloat16,
    attention_dtype=torch.bfloat16,
    moe_dtype=torch.bfloat16
)

model = Nucleus2MoE(config).to("cuda")
input = CollatedBatch(
    input=torch.randn((args.batch_size, args.t, args.height, args.width, 4), dtype=torch.float32, device="cuda"),
    target=None,
    sim_params_dict={},
    sim_params_tensor=torch.randn((args.batch_size, 14), dtype=torch.float32, device="cuda"),
    dx=torch.tensor(8, dtype=torch.float32, device="cuda"),
    dy=torch.tensor(8, dtype=torch.float32, device="cuda"),
    x_grid=None,
    y_grid=None
)

"""
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
torch._inductor.config.dce = True
torch._inductor.config.max_autotune_pointwise = True    
torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON,CPP,NVGEMM"
torch._inductor.config.max_autotune_gemm_search_space = "EXHAUSTIVE"
torch._inductor.config.shape_padding = True
torch._inductor.config.pad_outputs = True
torch._inductor.config.permute_fusion = True
torch._inductor.config.triton.divisible_by_16 = True
torch._inductor.config.triton.spill_threshold = 0 # disallow register spills
torch._inductor.config.triton.use_tensor_descriptor = True
torch._inductor.config.triton.enable_persistent_tma_matmul = True
torch._inductor.config.triton.enable_pdl = True
"""

if args.backward:
    def forward_and_backward():
        torch.compiler.cudagraph_mark_step_begin()
        output, _ = model(input)
        loss = output.sum()
        loss.backward()
    ms, min_ms, max_ms = do_bench(forward_and_backward, quantiles=[0.5, 0.2, 0.8])
else:
    def forward():
        torch.compiler.cudagraph_mark_step_begin()
        model(input) # compile before profile
    forward()
    ms, min_ms, max_ms = do_bench(forward, quantiles=[0.5, 0.2, 0.8])

print(torch.cuda.get_device_name(input.input.device), torch.cuda.get_device_capability(input.input.device))
if args.backward:
    print("Forward + Backward:")
else:
    print("Forward:")
print(f"{ms} ms")
print(f"{min_ms} ms")
print(f"{max_ms} ms")