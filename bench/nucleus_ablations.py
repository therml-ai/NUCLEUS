"""
Ablation benchmark: NUCLEUS vs attention/MLP variants.

Four full models compared:
  - NATTEN + MLP       (neighbor_vit)
  - Full Attn + MLP    (nucleus1_vit)
  - Full Attn + MoE    (nucleus1_vit_moe)
  - NUCLEUS            (nucleus1_moe  — NATTEN + MoE)

Two parameter-budget suites:

  Suite 1 — Matched total params
    Dense:  mlp_ratio = MLP_RATIO_DENSE
    MoE:    mlp_ratio = MLP_RATIO_DENSE / NUM_EXPERTS
            Total MoE MLP params ≈ dense total; active MoE params < dense.

  Suite 2 — Matched active params
    Dense:  mlp_ratio = MLP_RATIO_DENSE
    MoE:    mlp_ratio = MLP_RATIO_DENSE / TOPK
            Active MoE MLP params ≈ dense; total MoE = (NUM_EXPERTS / TOPK) × dense.

Metrics per model: total params, active params, GFLOPs, inference time,
                   train-step time, peak VRAM.
"""

import argparse
import csv
import statistics
from pathlib import Path

import torch
import torch._dynamo
from torch.utils.benchmark import Timer
from torch.utils.flop_counter import FlopCounterMode

from nucleus.data.batching import CollatedBatch
from nucleus.models import get_model
from nucleus.utils.parameter_count import count_model_parameters
from nucleus.layers.nucleus1_transformer_block import (
    Nucleus1TransformerBlock,
    Nucleus1TransformerNeighborBlock,
    Nucleus1TransformerMoEBlock,
    Nucleus1TransformerNeighborMoEBlock,
)
from nucleus.models.bubbleformer_vit import SpaceTimeBlock

torch._dynamo.config.cache_size_limit = 64

# ── Hardware / batch dims ────────────────────────────────────────────────────
B, T, H, W, C = 8, 5, 64, 64, 4
NUM_FLUID_PARAMS = 16
WARMUP_ITERS = 5
BENCHMARK_MIN_RUN_TIME = 5.0  # seconds for blocked_autorange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

# ── Shared backbone ──────────────────────────────────────────────────────────
BACKBONE = dict(
    patch_size=4,
    embed_dim=384,
    num_heads=6,
    processor_blocks=12,
)

COMMON = dict(
    input_fields=C,
    output_fields=C,
    num_fluid_params=NUM_FLUID_PARAMS,
)

# ── MoE topology ─────────────────────────────────────────────────────────────
NUM_EXPERTS = 8
TOPK = 2
LOAD_BAL_WT = 1e-5
MLP_RATIO_DENSE = 32.0

# Suite 1: total MoE MLP params ≈ dense total  →  ratio = dense / num_experts
MLP_RATIO_MOE_TOTAL = MLP_RATIO_DENSE / NUM_EXPERTS   # 4.0  (matches model_size_exp default)

# Suite 2: active MoE MLP params ≈ dense total  →  ratio = dense / topk
# Total MoE then = (num_experts / topk) × dense = 4× more params
MLP_RATIO_MOE_ACTIVE = MLP_RATIO_DENSE / TOPK         # 16.0

MOE_TOPOLOGY = dict(num_experts=NUM_EXPERTS, topk=TOPK, load_balance_loss_weight=LOAD_BAL_WT)

# ── Four models under comparison ─────────────────────────────────────────────
MODELS = [
    dict(model_name="neighbor_vit",       label="NATTEN + MLP",    moe=False),
    dict(model_name="nucleus1_vit",       label="Full Attn + MLP", moe=False),
    dict(model_name="nucleus1_vit_moe",   label="Full Attn + MoE", moe=True),
    dict(model_name="nucleus1_moe",       label="NUCLEUS",         moe=True),
    dict(model_name="bubbleformer_film_vit", label="Bubbleformer", moe=False,
         extra=dict(time_window=T, attn_scale=True, feat_scale=True)),
]

# ── Suite definitions ────────────────────────────────────────────────────────
SUITES = {
    "matched_total": dict(
        title="Suite 1 — Matched total params",
        description=(
            f"Dense mlp_ratio={MLP_RATIO_DENSE}, "
            f"MoE mlp_ratio={MLP_RATIO_MOE_TOTAL} "
            f"({NUM_EXPERTS} experts × {MLP_RATIO_MOE_TOTAL} ≈ dense {MLP_RATIO_DENSE}; "
            f"active MoE < dense)"
        ),
        mlp_ratio_dense=MLP_RATIO_DENSE,
        mlp_ratio_moe=MLP_RATIO_MOE_TOTAL,
    ),
    "matched_active": dict(
        title="Suite 2 — Matched active params",
        description=(
            f"Dense mlp_ratio={MLP_RATIO_DENSE}, "
            f"MoE mlp_ratio={MLP_RATIO_MOE_ACTIVE} "
            f"(active MoE MLP ≈ dense; total MoE = {NUM_EXPERTS // TOPK}× dense)"
        ),
        mlp_ratio_dense=MLP_RATIO_DENSE,
        mlp_ratio_moe=MLP_RATIO_MOE_ACTIVE,
    ),
}

# ── Shared input batch (param suites) ───────────────────────────────────────
batch = CollatedBatch(
    input=torch.randn(B, T, C, H, W, device=DEVICE),
    target=None,
    fluid_params_dict={},
    fluid_params_tensor=torch.randn(B, NUM_FLUID_PARAMS, device=DEVICE),
    x_grid=torch.linspace(0, 1, W, device=DEVICE),
    y_grid=torch.linspace(0, 1, H, device=DEVICE),
    dx=torch.tensor(1.0 / W, device=DEVICE),
    dy=torch.tensor(1.0 / H, device=DEVICE),
)

# ── Resolution scaling config ────────────────────────────────────────────────
# Block-level sweep: each block receives (B, T, H, W, embed_dim) directly.
# patch_grid = resolution // PATCH_SIZE_SCALE is the actual (H, W) fed to the block.
# Full-attention blocks will OOM at high resolutions — handled gracefully.
B_SCALE = 1
T_SCALE = 5
PATCH_SIZE_SCALE = 8
RESOLUTIONS = [64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]  # pixel resolutions

BACKBONE_SCALE = dict(
    embed_dim=256,
    num_heads=4,
)
MLP_RATIO_SCALE_DENSE = 16.0
MLP_RATIO_SCALE_MOE   = 2.0


class _BubbleformerBlockWrapper(torch.nn.Module):
    """SpaceTimeBlock expects (B, T, C, H, W); wrap to match (B, T, H, W, C)."""
    def __init__(self):
        super().__init__()
        self.block = SpaceTimeBlock(
            embed_dim=BACKBONE_SCALE["embed_dim"],
            num_heads=BACKBONE_SCALE["num_heads"],
            attn_scale=True,
            feat_scale=True,
            mlp_ratio=MLP_RATIO_SCALE_DENSE,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C) → (B, T, C, H, W) for SpaceTimeBlock → back
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = self.block(x)
        return x.permute(0, 1, 3, 4, 2).contiguous()


SCALE_BLOCKS = {
    "NATTEN + MLP":    lambda: Nucleus1TransformerNeighborBlock(
        BACKBONE_SCALE["embed_dim"], BACKBONE_SCALE["num_heads"], MLP_RATIO_SCALE_DENSE),
    "Full Attn + MLP": lambda: Nucleus1TransformerBlock(
        BACKBONE_SCALE["embed_dim"], BACKBONE_SCALE["num_heads"], MLP_RATIO_SCALE_DENSE),
    "Full Attn + MoE": lambda: Nucleus1TransformerMoEBlock(
        BACKBONE_SCALE["embed_dim"], BACKBONE_SCALE["num_heads"],
        NUM_EXPERTS, TOPK, LOAD_BAL_WT, MLP_RATIO_SCALE_MOE),
    "NUCLEUS":         lambda: Nucleus1TransformerNeighborMoEBlock(
        BACKBONE_SCALE["embed_dim"], BACKBONE_SCALE["num_heads"],
        NUM_EXPERTS, TOPK, LOAD_BAL_WT, MLP_RATIO_SCALE_MOE),
    "Bubbleformer":    lambda: _BubbleformerBlockWrapper(),
}


# ── Measurement helpers ──────────────────────────────────────────────────────
def _loss(out) -> torch.Tensor:
    """Extract a scalar loss from a model output (tensor or tuple)."""
    return out[0].sum() if isinstance(out, tuple) else out.sum()


def count_flops(model: torch.nn.Module, b: CollatedBatch) -> float:
    model.eval()
    with torch.no_grad():
        with FlopCounterMode(model, display=False) as fcm:
            model(b)
    return fcm.get_total_flops()


def measure_inference_ms(model, b: CollatedBatch) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        timer = Timer(stmt="model(b)", globals={"model": model, "b": b})
        for _ in range(WARMUP_ITERS):
            timer.timeit(1)
        m = timer.blocked_autorange(min_run_time=BENCHMARK_MIN_RUN_TIME)
    times_ms = [t * 1e3 for t in m.times]
    return m.mean * 1e3, statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0


def measure_vram_inference_mb(model, b: CollatedBatch) -> float:
    """Peak VRAM for a single forward pass (no backward)."""
    if DEVICE.type != "cuda":
        return float("nan")
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(b)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**2


def measure_train_ms(model) -> tuple[float, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    timer = Timer(
        stmt="""
out = model(batch)
loss = out[0].sum() if isinstance(out, tuple) else out.sum()
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
""",
        globals={"model": model, "batch": batch, "optimizer": optimizer},
    )
    for _ in range(WARMUP_ITERS):
        timer.timeit(1)
    m = timer.blocked_autorange(min_run_time=BENCHMARK_MIN_RUN_TIME)
    times_ms = [t * 1e3 for t in m.times]
    return m.mean * 1e3, statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0


def measure_vram_mb(model) -> float:
    if DEVICE.type != "cuda":
        return float("nan")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    out = model(batch)
    _loss(out).backward()
    optimizer.step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**2


# ── Per-model benchmark entry ─────────────────────────────────────────────────
def benchmark_model(suite_name: str, model_name: str, label: str, cfg: dict) -> dict:
    torch._dynamo.reset()
    model = get_model(model_name, **cfg).to(DEVICE)

    total_m  = count_model_parameters(model, active=False) / 1e6
    active_m = count_model_parameters(model, active=True)  / 1e6
    gflops   = count_flops(model, batch) / 1e9

    infer_ms, infer_std = measure_inference_ms(model, batch)
    train_ms, train_std = measure_train_ms(model)
    vram_mb             = measure_vram_mb(model)

    row = dict(
        suite=suite_name,
        model=model_name,
        label=label,
        active_m=active_m,
        total_m=total_m,
        gflops=gflops,
        inference_ms=infer_ms,
        inference_ms_std=infer_std,
        train_ms=train_ms,
        train_ms_std=train_std,
        vram_mb=vram_mb,
    )

    print(
        f"  {label:<22}  "
        f"active={active_m:>6.1f}M  total={total_m:>6.1f}M  "
        f"GFLOPs={gflops:>8.1f}  "
        f"infer={infer_ms:>7.1f}±{infer_std:>4.1f}ms  "
        f"train={train_ms:>7.1f}±{train_std:>4.1f}ms  "
        f"VRAM={vram_mb:>6.0f}MB"
    )

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return row


# ── Suite runners ─────────────────────────────────────────────────────────────
def run_suite(suite_key: str) -> list[dict]:
    suite = SUITES[suite_key]
    print(f"\n{'=' * 90}")
    print(suite["title"])
    print(suite["description"])
    print(
        f"backbone: embed={BACKBONE['embed_dim']}, blocks={BACKBONE['processor_blocks']}, "
        f"heads={BACKBONE['num_heads']}, patch={BACKBONE['patch_size']}  |  "
        f"input={C}ch  {H}×{W}  B={B}  T={T}"
    )
    print(f"{'=' * 90}")

    rows = []
    for m in MODELS:
        cfg = {**COMMON, **BACKBONE}
        cfg["mlp_ratio"] = suite["mlp_ratio_moe"] if m["moe"] else suite["mlp_ratio_dense"]
        if m["moe"]:
            cfg.update(MOE_TOPOLOGY)
        if m.get("extra"):
            cfg.update(m["extra"])
        rows.append(benchmark_model(suite_key, m["model_name"], m["label"], cfg))
    return rows


# ── Resolution scaling suite ─────────────────────────────────────────────────
def _count_flops_block(block: torch.nn.Module, x: torch.Tensor) -> float:
    block.eval()
    with torch.no_grad():
        with FlopCounterMode(block, display=False) as fcm:
            block(x)
    return fcm.get_total_flops()


def _measure_inference_ms_block(block: torch.nn.Module, x: torch.Tensor) -> tuple[float, float]:
    block.eval()
    with torch.no_grad():
        timer = Timer(stmt="block(x)", globals={"block": block, "x": x})
        for _ in range(WARMUP_ITERS):
            timer.timeit(1)
        m = timer.blocked_autorange(min_run_time=BENCHMARK_MIN_RUN_TIME)
    times_ms = [t * 1e3 for t in m.times]
    return m.mean * 1e3, statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0


def _measure_vram_inference_mb_block(block: torch.nn.Module, x: torch.Tensor) -> float:
    if DEVICE.type != "cuda":
        return float("nan")
    block.eval()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        block(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**2


def _measure_train_ms_block(block: torch.nn.Module, x: torch.Tensor) -> tuple[float, float]:
    block.train()
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-4)
    timer = Timer(
        stmt="""
out = block(x)
loss = out[0].sum() if isinstance(out, tuple) else out.sum()
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
""",
        globals={"block": block, "x": x, "optimizer": optimizer},
    )
    for _ in range(WARMUP_ITERS):
        timer.timeit(1)
    m = timer.blocked_autorange(min_run_time=BENCHMARK_MIN_RUN_TIME)
    times_ms = [t * 1e3 for t in m.times]
    return m.mean * 1e3, statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0


def _measure_vram_train_mb_block(block: torch.nn.Module, x: torch.Tensor) -> float:
    if DEVICE.type != "cuda":
        return float("nan")
    block.train()
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-4)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    block.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    out = block(x)
    loss = out[0].sum() if isinstance(out, tuple) else out.sum()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**2


def run_resolution_sweep() -> list[dict]:
    print(f"\n{'=' * 90}")
    print("Suite 3 — Resolution scaling (single block)")
    print(
        f"B={B_SCALE}, T={T_SCALE}, embed={BACKBONE_SCALE['embed_dim']}, "
        f"heads={BACKBONE_SCALE['num_heads']}, patch_size={PATCH_SIZE_SCALE}  |  "
        f"dense mlp_ratio={MLP_RATIO_SCALE_DENSE}, MoE mlp_ratio={MLP_RATIO_SCALE_MOE}"
    )
    print(f"{'=' * 90}")

    rows = []
    for res in RESOLUTIONS:
        patch_grid = res // PATCH_SIZE_SCALE
        # blocks take (B, T, H, W, embed_dim)
        x = torch.randn(B_SCALE, T_SCALE, patch_grid, patch_grid, BACKBONE_SCALE["embed_dim"], device=DEVICE)
        print(f"\n  resolution={res}×{res}px  (patch grid={patch_grid}×{patch_grid}, patch_size={PATCH_SIZE_SCALE})")

        for label, build_fn in SCALE_BLOCKS.items():
            oom = False
            try:
                block = build_fn().to(DEVICE)
                total_m  = count_model_parameters(block, active=False) / 1e6
                active_m = count_model_parameters(block, active=True)  / 1e6
                gflops   = _count_flops_block(block, x) / 1e9
                infer_ms, infer_std = _measure_inference_ms_block(block, x)
                train_ms, train_std = _measure_train_ms_block(block, x)
                train_vram_mb       = _measure_vram_train_mb_block(block, x)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    oom = True
                    total_m = active_m = gflops = float("nan")
                    infer_ms = infer_std = train_ms = train_std = train_vram_mb = float("nan")
                else:
                    raise
            finally:
                if "block" in dir():
                    del block
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            infer_status = "OOM" if oom else f"{infer_ms:>7.1f}±{infer_std:>4.1f}ms"
            train_status = "OOM" if oom else f"{train_ms:>7.1f}±{train_std:>4.1f}ms"
            print(
                f"    {label:<22}  "
                f"active={active_m:>6.1f}M  total={total_m:>6.1f}M  "
                f"GFLOPs={gflops:>8.1f}  infer={infer_status}  "
                f"train={train_status}  VRAM={train_vram_mb:>6.0f}MB"
            )

            rows.append(dict(
                suite="resolution_scaling",
                resolution=res,
                patch_grid=patch_grid,
                label=label,
                active_m=active_m,
                total_m=total_m,
                gflops=gflops,
                inference_ms=infer_ms,
                inference_ms_std=infer_std,
                train_ms=train_ms,
                train_ms_std=train_std,
                train_vram_mb=train_vram_mb,
                oom=oom,
            ))

    return rows


# ── Output (param suites) ─────────────────────────────────────────────────────
FIELDNAMES = [
    "suite", "model", "label",
    "active_m", "total_m", "gflops",
    "inference_ms", "inference_ms_std",
    "train_ms", "train_ms_std",
    "vram_mb",
]

SCALING_FIELDNAMES = [
    "suite", "resolution", "patch_grid", "label",
    "active_m", "total_m", "gflops",
    "inference_ms", "inference_ms_std",
    "train_ms", "train_ms_std",
    "train_vram_mb", "oom",
]


def write_csv(rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nucleus_ablations.csv"
    with out_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_scaling_csv(rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nucleus_scaling.csv"
    with out_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=SCALING_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def print_table(rows: list[dict]) -> None:
    col_suite  = 16
    col_label  = 22
    header = (
        f"{'Suite':<{col_suite}} {'Model':<{col_label}} "
        f"{'Active(M)':>9} {'Total(M)':>9} {'GFLOPs':>8} "
        f"{'Infer(ms)':>14} {'Train(ms)':>14} {'VRAM(MB)':>9}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)

    prev_suite = None
    for r in rows:
        if r["suite"] != prev_suite:
            print(sep)
            prev_suite = r["suite"]
        print(
            f"{r['suite']:<{col_suite}} {r['label']:<{col_label}} "
            f"{r['active_m']:>9.1f} {r['total_m']:>9.1f} {r['gflops']:>8.1f} "
            f"{r['inference_ms']:>8.1f}±{r['inference_ms_std']:>4.1f} "
            f"{r['train_ms']:>8.1f}±{r['train_ms_std']:>4.1f} "
            f"{r['vram_mb']:>9.0f}"
        )
    print(sep)


def print_scaling_table(rows: list[dict]) -> None:
    col_label = 22
    header = (
        f"{'Res(px)':>8} {'Block':<{col_label}} "
        f"{'Active(M)':>9} {'Total(M)':>9} {'GFLOPs':>8} "
        f"{'Infer(ms)':>14} {'Train(ms)':>14} {'TrainVRAM(MB)':>13}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)

    prev_grid = None
    for r in rows:
        if r["patch_grid"] != prev_grid:
            print(sep)
            prev_grid = r["patch_grid"]
        if r["oom"]:
            infer_str = f"{'OOM':>14}"
            train_str = f"{'OOM':>14}"
            vram_str  = f"{'OOM':>13}"
        else:
            infer_str = f"{r['inference_ms']:>8.1f}±{r['inference_ms_std']:>4.1f}"
            train_str = f"{r['train_ms']:>8.1f}±{r['train_ms_std']:>4.1f}"
            vram_str  = f"{r['train_vram_mb']:>13.0f}"
        active_str = "nan" if r["active_m"] != r["active_m"] else f"{r['active_m']:>9.1f}"
        gflops_str = "nan" if r["gflops"] != r["gflops"] else f"{r['gflops']:>8.1f}"
        print(
            f"{r['resolution']:>8} {r['label']:<{col_label}} "
            f"{active_str:>9} {r['total_m'] if r['total_m']==r['total_m'] else float('nan'):>9.1f} "
            f"{gflops_str:>8} {infer_str} {train_str} {vram_str}"
        )
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for CSV output")
    parser.add_argument(
        "--suite",
        choices=["all", "matched_total", "matched_active", "resolution_scaling"],
        default="all",
        help="Which suite(s) to run (default: all)",
    )
    args = parser.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir else Path.home() / "temp") / "nucleus_ablations"
    print(f"Output dir: {out_dir.resolve()}")

    param_rows: list[dict] = []
    if args.suite in ("all", "matched_total"):
        param_rows.extend(run_suite("matched_total"))
    if args.suite in ("all", "matched_active"):
        param_rows.extend(run_suite("matched_active"))
    if param_rows:
        print_table(param_rows)
        out_path = write_csv(param_rows, out_dir)
        print(f"\nWrote param-suite CSV to {out_path}")

    if args.suite in ("all", "resolution_scaling"):
        scaling_rows = run_resolution_sweep()
        print_scaling_table(scaling_rows)
        out_path = write_scaling_csv(scaling_rows, out_dir)
        print(f"\nWrote scaling CSV to {out_path}")


if __name__ == "__main__":
    main()
