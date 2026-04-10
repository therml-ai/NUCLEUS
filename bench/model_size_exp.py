import argparse
import csv
import statistics
from pathlib import Path

import torch
import torch._dynamo
from torch.utils.benchmark import Timer

from nucleus.data.batching import CollatedBatch
from nucleus.models import get_model
from nucleus.utils.parameter_count import count_model_parameters

torch._dynamo.config.cache_size_limit = 64


# Shared benchmark setup
B, T, H, W, C = 10, 5, 64, 64, 4
NUM_FLUID_PARAMS = 16
BENCHMARK_REPEATS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


# All benchmark configs live here
COMMON_MODEL_CONFIG = dict(
    input_fields=C,
    output_fields=C,
    num_fluid_params=NUM_FLUID_PARAMS,
)

MATCHED_BACKBONE_CONFIGS = {
    "small": dict(
        patch_size=4,
        embed_dim=256,
        processor_blocks=6,
        num_heads=4,
    ),
    "large": dict(
        patch_size=4,
        embed_dim=512,
        processor_blocks=8,
        num_heads=8,
    ),
}

MATCHED_VARIANTS = [
    dict(model_name="neighbor_vit", label="NATTEN + MLP", moe=False, benchmark=True),
    dict(model_name="nucleus1_vit_moe", label="Full Attn + MoE", moe=True, benchmark=True),
    dict(model_name="nucleus1_moe", label="NUCLEUS", moe=True, benchmark=True),
    dict(model_name="bubbleformer_film_vit", label="Bubbleformer", moe=False, benchmark=True),
]

MATCHED_MOE_CONFIG = dict(
    num_experts=8,
    topk=2,
    load_balance_loss_weight=0.01,
)

# These are the fixed big/small model configs from the earlier benchmark.
EXACT_MODEL_CONFIGS = [
    dict(
        size="small",
        model_name="neighbor_vit",
        label="NATTEN + MLP",
        benchmark=True,
        cfg=dict(
            patch_size=4,
            embed_dim=384,
            processor_blocks=12,
            num_heads=6,
            mlp_ratio=32.0,
        ),
    ),
    dict(
        size="small",
        model_name="nucleus1_vit_moe",
        label="Full Attn + MoE",
        benchmark=True,
        cfg=dict(
            patch_size=4,
            embed_dim=384,
            processor_blocks=12,
            num_heads=6,
            num_experts=8,
            topk=2,
            load_balance_loss_weight=1e-5,
        ),
    ),
    dict(
        size="small",
        model_name="nucleus1_moe",
        label="NUCLEUS",
        benchmark=True,
        cfg=dict(
            patch_size=4,
            embed_dim=384,
            processor_blocks=12,
            num_heads=6,
            num_experts=8,
            topk=2,
            load_balance_loss_weight=1e-5,
        ),
    ),
    dict(
        size="small",
        model_name="bubbleformer_film_vit",
        label="Bubbleformer",
        benchmark=True,
        cfg=dict(
            patch_size=4,
            embed_dim=384,
            processor_blocks=12,
            num_heads=6,
            time_window=T,
            attn_scale=True,
            feat_scale=True,
            mlp_ratio=32.0,
        ),
    ),
    dict(
        size="large",
        model_name="neighbor_vit",
        label="NATTEN + MLP",
        benchmark=True,
        cfg=dict(
            patch_size=4,
            embed_dim=384,
            processor_blocks=12,
            num_heads=6,
            mlp_ratio=32.0,
        ),
    ),
    dict(
        size="large",
        model_name="nucleus1_vit_moe",
        label="Full Attn + MoE",
        benchmark=True,
        cfg=dict(
            patch_size=4,
            embed_dim=384,
            processor_blocks=12,
            num_heads=6,
            num_experts=8,
            topk=2,
            load_balance_loss_weight=1e-5,
        ),
    ),
    dict(
        size="large",
        model_name="nucleus1_moe",
        label="NUCLEUS",
        benchmark=True,
        cfg=dict(
            patch_size=4,
            embed_dim=384,
            processor_blocks=12,
            num_heads=6,
            num_experts=8,
            topk=2,
            load_balance_loss_weight=1e-5,
        ),
    ),
    dict(
        size="large",
        model_name="bubbleformer_film_vit",
        label="Bubbleformer",
        benchmark=True,
        cfg=dict(
            patch_size=4,
            embed_dim=384,
            processor_blocks=12,
            num_heads=6,
            time_window=T,
            attn_scale=True,
            feat_scale=True,
            mlp_ratio=32.0,
        ),
    ),
]


# Shared inputs
tchw_batch = CollatedBatch(
    input=torch.randn(B, T, C, H, W, device=device),
    target=None,
    fluid_params_dict={},
    fluid_params_tensor=torch.randn(B, NUM_FLUID_PARAMS, device=device),
    x_grid=torch.linspace(1, 1, W, device=device),
    y_grid=torch.linspace(1, 1, H, device=device),
    dx=torch.tensor(1.0 / W, device=device),
    dy=torch.tensor(1.0 / H, device=device),
)


# Helpers
def measure_inference_ms(model, batch, repeats=BENCHMARK_REPEATS) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        timer = Timer(stmt="model(batch)", globals={"model": model, "batch": batch})
        times_ms = [timer.timeit(1).mean * 1e3 for _ in range(repeats)]
    return statistics.mean(times_ms), statistics.stdev(times_ms)


def measure_train_step_ms(model, batch, repeats=BENCHMARK_REPEATS) -> tuple[float, float]:
    model.train()
    timer = Timer(
        stmt="""
out = model(batch)
loss = out[0].sum() if isinstance(out, tuple) else out.sum()
loss.backward()
model.zero_grad(set_to_none=True)
""",
        globals={"model": model, "batch": batch},
    )
    times_ms = [timer.timeit(1).mean * 1e3 for _ in range(repeats)]
    return statistics.mean(times_ms), statistics.stdev(times_ms)


def measure_vram_mb(model, batch):
    if device.type != "cuda":
        return float("nan")
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    out = model(batch)
    loss = out[0].sum() if isinstance(out, tuple) else out.sum()
    loss.backward()
    model.zero_grad(set_to_none=True)
    return torch.cuda.max_memory_allocated(device) / 1024**2


def benchmark_model(suite: str, size_name: str, model_name: str, label: str, cfg: dict, do_benchmark: bool) -> dict:
    torch._dynamo.reset()
    model = get_model(model_name, **cfg).to(device)

    total = count_model_parameters(model, active=False)
    active = count_model_parameters(model, active=True)

    row = {
        "suite": suite,
        "size": size_name,
        "model": model_name,
        "label": label,
        "active_m": active / 1e6,
        "total_m": total / 1e6,
        "inference_ms": float("nan"),
        "inference_ms_std": float("nan"),
        "train_ms": float("nan"),
        "train_ms_std": float("nan"),
        "vram_mb": float("nan"),
    }

    print(f"\n  [{label}] -> {model_name}")
    print(f"  {'active':<10} {row['active_m']:>6.1f}M")
    print(f"  {'total':<10} {row['total_m']:>6.1f}M")

    if do_benchmark:
        row["inference_ms"], row["inference_ms_std"] = measure_inference_ms(model, tchw_batch)
        row["train_ms"], row["train_ms_std"] = measure_train_step_ms(model, tchw_batch)
        row["vram_mb"] = measure_vram_mb(model, tchw_batch)
        print(f"  {'inference':<10} {row['inference_ms']:>6.1f} ± {row['inference_ms_std']:>4.1f} ms")
        print(f"  {'train step':<10} {row['train_ms']:>6.1f} ± {row['train_ms_std']:>4.1f} ms")
        print(f"  {'VRAM':<10} {row['vram_mb']:>6.0f} MB")
    else:
        print("  (timing skipped)")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return row


def write_csv(rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_benchmark.csv"
    fieldnames = [
        "suite",
        "size",
        "model",
        "label",
        "active_m",
        "total_m",
        "inference_ms",
        "inference_ms_std",
        "train_ms",
        "train_ms_std",
        "vram_mb",
    ]
    with out_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def run_matched_backbone_suite() -> list[dict]:
    print("=== Matched ablation benchmark ===")
    print("Each size uses one shared backbone config across all compared variants.")

    rows = []
    for size_name, base_cfg in MATCHED_BACKBONE_CONFIGS.items():
        print(f"\n{'-' * 60}")
        print(
            f"Size: {size_name} | patch={base_cfg['patch_size']} embed={base_cfg['embed_dim']} "
            f"blocks={base_cfg['processor_blocks']} heads={base_cfg['num_heads']} "
            f"input/output={C} resolution={H}x{W} batch={B}x{T}"
        )
        print(f"{'-' * 60}")

        for variant in MATCHED_VARIANTS:
            cfg = {**COMMON_MODEL_CONFIG, **base_cfg}
            if variant["moe"]:
                cfg.update(MATCHED_MOE_CONFIG)
            if variant["model_name"] == "bubbleformer_film_vit":
                cfg.update(
                    time_window=T,
                    attn_scale=True,
                    feat_scale=True,
                )
            rows.append(
                benchmark_model(
                    suite="matched_backbone",
                    size_name=size_name,
                    model_name=variant["model_name"],
                    label=variant["label"],
                    cfg=cfg,
                    do_benchmark=variant["benchmark"],
                )
            )
    return rows


def run_exact_model_suite() -> list[dict]:
    print(f"\n=== Exact model benchmark ===")
    print("Fixed small/big configs for NUCLEUS and Bubbleformer baselines.")

    rows = []
    for entry in EXACT_MODEL_CONFIGS:
        cfg = {**COMMON_MODEL_CONFIG, **entry["cfg"]}
        rows.append(
            benchmark_model(
                suite="exact_models",
                size_name=entry["size"],
                model_name=entry["model_name"],
                label=entry["label"],
                cfg=cfg,
                do_benchmark=entry["benchmark"],
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = (Path(args.out_dir) if args.out_dir else Path.home() / "temp") / "ablation_report"

    rows = []
    rows.extend(run_matched_backbone_suite())
    rows.extend(run_exact_model_suite())

    out_path = write_csv(rows, out_dir)
    print(f"\nWrote benchmark CSV to {out_path}")


if __name__ == "__main__":
    main()
