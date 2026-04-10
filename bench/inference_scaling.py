"""
Inference-only scaling sweep for the four models in problem_size_exp,
extended to large resolutions (up to 2048).

Only measures forward pass time and peak VRAM — no backward pass.
"""

import argparse
import statistics

import torch
import torch._dynamo
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from torch.utils.benchmark import Timer

from nucleus.data.batching import CollatedBatch
from nucleus.models import get_model

torch._dynamo.config.cache_size_limit = 64

#CONFIG
BATCH_SIZE       = 1
TIME_WINDOW      = 5
CHANNELS         = 4
NUM_FLUID_PARAMS = 16
RESOLUTIONS      = [64, 128, 256, 512, 768, 1024, 1536, 2048]
BENCHMARK_REPEATS = 20

COMMON_MODEL_CONFIG = dict(
    input_fields=CHANNELS,
    output_fields=CHANNELS,
    num_fluid_params=NUM_FLUID_PARAMS,
    patch_size=8,
    embed_dim=384,
    processor_blocks=4,
    num_heads=6,
)

MODELS = [
    dict(model_name="neighbor_vit",      label="NATTEN + MLP",    cfg=dict(mlp_ratio=4.0)),
    dict(model_name="nucleus1_vit_moe",  label="Full Attn + MoE", cfg=dict(num_experts=8, topk=2, load_balance_loss_weight=1e-5)),
    dict(model_name="nucleus1_moe",      label="NUCLEUS",         cfg=dict(num_experts=8, topk=2, load_balance_loss_weight=1e-5)),
    dict(model_name="bubbleformer_film_vit", label="Bubbleformer", cfg=dict(time_window=TIME_WINDOW, attn_scale=True, feat_scale=True, mlp_ratio=4.0)),
]

STYLES = {
    "NATTEN + MLP":    dict(color="#2ecc71", linestyle="-",  marker="o"),
    "Full Attn + MoE": dict(color="#e05c5c", linestyle="-",  marker="s"),
    "NUCLEUS":         dict(color="#2980b9", linestyle="-",  marker="^"),
    "Bubbleformer":    dict(color="#9b59b6", linestyle="-",  marker="D"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------------------------


def make_batch(resolution: int) -> CollatedBatch:
    return CollatedBatch(
        input=torch.randn(BATCH_SIZE, TIME_WINDOW, CHANNELS, resolution, resolution, device=DEVICE),
        target=None,
        fluid_params_dict={},
        fluid_params_tensor=torch.randn(BATCH_SIZE, NUM_FLUID_PARAMS, device=DEVICE),
        x_grid=torch.linspace(0, 1, resolution, device=DEVICE),
        y_grid=torch.linspace(0, 1, resolution, device=DEVICE),
        dx=torch.tensor(1.0 / resolution, device=DEVICE),
        dy=torch.tensor(1.0 / resolution, device=DEVICE),
    )


def measure_inference(model, batch) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        timer = Timer(stmt="model(batch)", globals={"model": model, "batch": batch})
        times_ms = [timer.timeit(1).mean * 1e3 for _ in range(BENCHMARK_REPEATS)]
    return statistics.mean(times_ms), statistics.stdev(times_ms)


def measure_vram(model, batch) -> float:
    if DEVICE.type != "cuda":
        return float("nan")
    model.eval()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    with torch.no_grad():
        model(batch)
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**2


def run_sweep():
    # results[label] = {"ms": [...], "ms_std": [...], "vram_mb": [...], "resolutions": [...]}
    results = {m["label"]: {"ms": [], "ms_std": [], "vram_mb": [], "resolutions": []} for m in MODELS}

    for resolution in RESOLUTIONS:
        print(f"\n--- {resolution}×{resolution} ---")
        batch = make_batch(resolution)

        for model_spec in MODELS:
            label = model_spec["label"]
            cfg   = {**COMMON_MODEL_CONFIG, **model_spec["cfg"]}

            torch._dynamo.reset()
            model = get_model(model_spec["model_name"], **cfg).to(DEVICE)
            ms, ms_std = measure_inference(model, batch)
            vram_mb = measure_vram(model, batch)
            results[label]["ms"].append(ms)
            results[label]["ms_std"].append(ms_std)
            results[label]["vram_mb"].append(vram_mb)
            results[label]["resolutions"].append(resolution)
            print(f"  {label:<22}  {ms:>8.1f} ± {ms_std:>5.1f} ms   {vram_mb:>7.0f} MB")
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    return results


def make_plots(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Inference scaling  |  B={BATCH_SIZE}, T={TIME_WINDOW}, "
        f"embed=384, heads=6, blocks=4, patch=8",
        fontsize=10,
    )

    for ax_idx, (ax, metric, ylabel, title) in enumerate(zip(
        axes,
        ["ms",      "vram_mb"],
        ["Wall time (ms)", "Peak VRAM (MB)"],
        ["Inference time vs resolution", "Peak VRAM vs resolution"],
    )):
        for label, data in results.items():
            if not data["resolutions"]:
                continue
            style = STYLES[label]
            ax.plot(data["resolutions"], data[metric],
                    label=label, **style, linewidth=2, markersize=6)
            if metric == "ms":
                ms = data["ms"]
                std = data["ms_std"]
                ax.fill_between(data["resolutions"],
                                [m - s for m, s in zip(ms, std)],
                                [m + s for m, s in zip(ms, std)],
                                color=style["color"], alpha=0.15)

        ax.set_xlabel("Resolution (px)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(RESOLUTIONS)
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(r) for r in RESOLUTIONS]))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "inference_scaling.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = (Path(args.out_dir) if args.out_dir else Path.home() / "temp") / "inference_scaling"

    print(f"Device: {DEVICE}")
    results = run_sweep()
    make_plots(results, out_dir)
