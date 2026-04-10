"""
Profile FLOPs and runtime for individual attention/MoE blocks,
swept over a range of spatial patch grid sizes.

SPATIAL_SIZES is a list of (H, W) patch grid dimensions.
Set image_resolution = H * patch_size to get the pixel resolution.
All other config lives in the INPUT CONFIG section.
"""

import argparse
import statistics

import torch
import torch.nn as nn
from torch.utils.benchmark import Timer
from torch.utils.flop_counter import FlopCounterMode
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from nucleus.layers.nucleus1_transformer_block import (
    Nucleus1TransformerMoEBlock,
    Nucleus1TransformerNeighborMoEBlock,
    Nucleus1TransformerAxialMoEBlock,
)
from nucleus.layers.nucleus1_space_time_attention import (
    Nucleus1SpaceTimeAttention,
    Nucleus1SpaceTimeNeighborAttention,
    Nucleus1SpaceTimeAxialAttention,
)

#CONFIG
BATCH       = 1
TIME        = 5
EMBED_DIM   = 384
NUM_HEADS   = 6
NUM_EXPERTS = 8
TOPK        = 2
MLP_RATIO   = 1.0
LOAD_BAL_WT = 1e-5

# Patch grid sizes to sweep (H, W). Assumes square grids.
# Equivalent pixel resolution = size * patch_size (e.g. patch_size=8 → 8=64px, 16=128px, ...)
PATCH_SIZE   = 8   # only used for axis labels
SPATIAL_SIZES = [8, 16, 32, 64, 96, 128, 256]  # patch grid side length

BENCHMARK_REPEATS = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODULES = {
    "Full Attn (attn only)":   lambda: Nucleus1SpaceTimeAttention(EMBED_DIM, NUM_HEADS),
    "Axial (attn only)":       lambda: Nucleus1SpaceTimeAxialAttention(EMBED_DIM, NUM_HEADS),
    "NATTEN (attn only)":      lambda: Nucleus1SpaceTimeNeighborAttention(EMBED_DIM, NUM_HEADS),
    "Full Attn + MoE (block)": lambda: Nucleus1TransformerMoEBlock(
        EMBED_DIM, NUM_HEADS, NUM_EXPERTS, TOPK, LOAD_BAL_WT, MLP_RATIO),
    "Axial + MoE (block)":     lambda: Nucleus1TransformerAxialMoEBlock(
        EMBED_DIM, NUM_HEADS, NUM_EXPERTS, TOPK, LOAD_BAL_WT, MLP_RATIO),
    "NATTEN + MoE (block)":    lambda: Nucleus1TransformerNeighborMoEBlock(
        EMBED_DIM, NUM_HEADS, NUM_EXPERTS, TOPK, LOAD_BAL_WT, MLP_RATIO),
}

# Colors / markers for each module
STYLES = {
    "Full Attn (attn only)":   dict(color="#e05c5c", linestyle="--", marker="o"),
    "Axial (attn only)":       dict(color="#27ae60", linestyle="--", marker="^"),
    "NATTEN (attn only)":      dict(color="#5c9ee0", linestyle="--", marker="s"),
    "Full Attn + MoE (block)": dict(color="#c0392b", linestyle="-",  marker="o"),
    "Axial + MoE (block)":     dict(color="#1e8449", linestyle="-",  marker="^"),
    "NATTEN + MoE (block)":    dict(color="#2980b9", linestyle="-",  marker="s"),
}


def count_flops(module: nn.Module, x: torch.Tensor) -> float:
    module.eval()
    with torch.no_grad():
        with FlopCounterMode(module, display=False) as fcm:
            module(x)
    return fcm.get_total_flops()


def benchmark_ms(module: nn.Module, x: torch.Tensor) -> tuple[float, float]:
    module.eval()
    with torch.no_grad():
        timer = Timer(stmt="module(x)", globals={"module": module, "x": x})
        times_ms = [timer.timeit(1).mean * 1e3 for _ in range(BENCHMARK_REPEATS)]
    return statistics.mean(times_ms), statistics.stdev(times_ms)


def run_sweep():
    results = {name: {"flops": [], "ms": [], "ms_std": []} for name in MODULES}
    spatial_patch_counts = []

    for size in SPATIAL_SIZES:
        h = w = size
        spatial_patch_counts.append(size * PATCH_SIZE)
        x = torch.randn(BATCH, TIME, h, w, EMBED_DIM, device=DEVICE)
        px = size * PATCH_SIZE

        print(f"\n--- {size}×{size} patches ({px}×{px}px) ---")

        for name, build_fn in MODULES.items():
            module = build_fn().to(DEVICE)
            flops = count_flops(module, x)
            ms, ms_std = benchmark_ms(module, x)
            results[name]["flops"].append(flops / 1e9)   # store as GFLOPs
            results[name]["ms"].append(ms)
            results[name]["ms_std"].append(ms_std)
            del module
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  {name:<35}  {flops/1e9:>7.2f} GFLOPs   {ms:>7.2f} ± {ms_std:>5.2f} ms")

    return results, spatial_patch_counts


def make_plots(results, spatial_patch_counts, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    resolutions = [s * PATCH_SIZE for s in SPATIAL_SIZES]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Block scaling  |  B={BATCH}, T={TIME}, embed={EMBED_DIM}, "
        f"heads={NUM_HEADS}, experts={NUM_EXPERTS}, topk={TOPK}, mlp_ratio={MLP_RATIO}",
        fontsize=10,
    )

    def setup_xaxis(ax):
        ax.set_xscale("log")
        ax.set_xticks(resolutions)
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(r) for r in resolutions]))
        ax.set_xlabel("Image resolution (px)")

    # ---- FLOPs ----
    ax = axes[0]
    for name, data in results.items():
        ax.plot(spatial_patch_counts, data["flops"], label=name, **STYLES[name], linewidth=2, markersize=6)
    setup_xaxis(ax)
    ax.set_ylabel("GFLOPs (forward pass)")
    ax.set_title("FLOPs vs resolution")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # ---- Runtime ----
    ax = axes[1]
    for name, data in results.items():
        style = STYLES[name]
        xs = spatial_patch_counts
        ms = data["ms"]
        std = data["ms_std"]
        ax.plot(xs, ms, label=name, **style, linewidth=2, markersize=6)
        ax.fill_between(xs,
                        [m - s for m, s in zip(ms, std)],
                        [m + s for m, s in zip(ms, std)],
                        color=style["color"], alpha=0.15)
    setup_xaxis(ax)
    ax.set_ylabel("Wall time (ms)")
    ax.set_title("Runtime vs resolution")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "block_scaling.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.show()


def sanity_check():
    full_block   = Nucleus1TransformerMoEBlock(EMBED_DIM, NUM_HEADS, NUM_EXPERTS, TOPK, LOAD_BAL_WT, MLP_RATIO)
    axial_block  = Nucleus1TransformerAxialMoEBlock(EMBED_DIM, NUM_HEADS, NUM_EXPERTS, TOPK, LOAD_BAL_WT, MLP_RATIO)
    natten_block = Nucleus1TransformerNeighborMoEBlock(EMBED_DIM, NUM_HEADS, NUM_EXPERTS, TOPK, LOAD_BAL_WT, MLP_RATIO)
    fa = type(full_block.attention.spatial).__name__
    ax = type(axial_block.attention.spatial).__name__
    na = type(natten_block.attention.spatial).__name__
    assert len({fa, ax, na}) == 3, f"Expected 3 distinct spatial attn types, got {fa}, {ax}, {na}"
    print(f"Sanity check OK: full={fa}, axial={ax}, natten={na}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = (Path(args.out_dir) if args.out_dir else Path.home() / "temp") / "block_flop_profile"

    print(f"Device: {DEVICE}")
    sanity_check()
    results, spatial_patch_counts = run_sweep()
    make_plots(results, spatial_patch_counts, out_dir)
