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


# Experiment config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

BATCH_SIZE = 1
TIME_WINDOW = 5
CHANNELS = 4
NUM_FLUID_PARAMS = 16
RESOLUTIONS = [512, 256, 128, 64]
BENCHMARK_REPEATS = 20

# Shared backbone used by both experiments.
COMMON_MODEL_CONFIG = dict(
    input_fields=CHANNELS,
    output_fields=CHANNELS,
    num_fluid_params=NUM_FLUID_PARAMS,
    patch_size=8,
    embed_dim=384,
    processor_blocks=4,
    num_heads=6,
)

# Two experiment definitions live here so you can edit them in one place.
EXPERIMENTS = [
    dict(
        name="mlp_ratio_32",
        description="Keep the shared backbone fixed, but widen dense MLP models to mlp_ratio=32.",
        models=[
            dict(
                model_name="neighbor_vit",
                label="NATTEN + MLP",
                cfg=dict(mlp_ratio=32.0),
                is_moe=False,
            ),
            dict(
                model_name="nucleus1_vit_moe",
                label="Full Attn + MoE",
                cfg=dict(num_experts=8, topk=2, load_balance_loss_weight=1e-5),
                is_moe=True,
            ),
            dict(
                model_name="nucleus1_moe",
                label="NUCLEUS",
                cfg=dict(num_experts=8, topk=2, load_balance_loss_weight=1e-5),
                is_moe=True,
            ),
            dict(
                model_name="bubbleformer_film_vit",
                label="Bubbleformer",
                cfg=dict(
                    time_window=TIME_WINDOW,
                    attn_scale=True,
                    feat_scale=True,
                    mlp_ratio=32.0,
                ),
                is_moe=False,
            ),
        ],
    ),
    dict(
        name="shared_backbone",
        description="Keep the actual model configs fixed and sweep only spatial resolution.",
        models=[
            dict(
                model_name="neighbor_vit",
                label="NATTEN + MLP",
                cfg=dict(mlp_ratio=4.0),
                is_moe=False,
            ),
            dict(
                model_name="nucleus1_vit_moe",
                label="Full Attn + MoE",
                cfg=dict(num_experts=8, topk=2, load_balance_loss_weight=1e-5),
                is_moe=True,
            ),
            dict(
                model_name="nucleus1_moe",
                label="NUCLEUS",
                cfg=dict(num_experts=8, topk=2, load_balance_loss_weight=1e-5),
                is_moe=True,
            ),
            dict(
                model_name="bubbleformer_film_vit",
                label="Bubbleformer",
                cfg=dict(
                    time_window=TIME_WINDOW,
                    attn_scale=True,
                    feat_scale=True,
                    mlp_ratio=4.0,
                ),
                is_moe=False,
            ),
        ],
    ),
]


# Batch + metrics helpers
def make_batch(resolution: int) -> CollatedBatch:
    return CollatedBatch(
        input=torch.randn(BATCH_SIZE, TIME_WINDOW, CHANNELS, resolution, resolution, device=DEVICE),
        target=None,
        fluid_params_dict={},
        fluid_params_tensor=torch.randn(BATCH_SIZE, NUM_FLUID_PARAMS, device=DEVICE),
        x_grid=torch.linspace(1, 1, resolution, device=DEVICE),
        y_grid=torch.linspace(1, 1, resolution, device=DEVICE),
        dx=torch.tensor(1.0 / resolution, device=DEVICE),
        dy=torch.tensor(1.0 / resolution, device=DEVICE),
    )


def tokens_per_sample(resolution: int, patch_size: int) -> int:
    return TIME_WINDOW * (resolution // patch_size) * (resolution // patch_size)


def pixels_per_batch(resolution: int) -> int:
    return BATCH_SIZE * TIME_WINDOW * resolution * resolution


def scalar_loss(output):
    if isinstance(output, tuple):
        pred, moe_outputs = output
        loss = pred.sum()
        if moe_outputs:
            for moe_output in moe_outputs:
                if hasattr(moe_output, "load_balance_loss"):
                    loss = loss + moe_output.load_balance_loss
                elif hasattr(moe_output, "router_output"):
                    loss = loss + moe_output.router_output.load_balance_loss
        return loss
    return output.sum()


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
loss = scalar_loss(out)
loss.backward()
model.zero_grad(set_to_none=True)
""",
        globals={"model": model, "batch": batch, "scalar_loss": scalar_loss},
    )
    times_ms = [timer.timeit(1).mean * 1e3 for _ in range(repeats)]
    return statistics.mean(times_ms), statistics.stdev(times_ms)


def measure_inference_vram_mb(model, batch) -> float:
    if DEVICE.type != "cuda":
        return float("nan")
    model.eval()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    with torch.no_grad():
        model(batch)
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**2


def measure_train_vram_mb(model, batch) -> float:
    if DEVICE.type != "cuda":
        return float("nan")
    model.train()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    out = model(batch)
    loss = scalar_loss(out)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**2


def extract_moe_metrics(output) -> dict:
    metrics = {
        "mean_tokens_per_expert": float("nan"),
        "mean_load_imbalance": float("nan"),
        "mean_active_experts_frac": float("nan"),
    }
    if not isinstance(output, tuple):
        return metrics

    _, moe_outputs = output
    if not moe_outputs:
        return metrics

    load_imbalances = []
    active_expert_fracs = []
    mean_tokens = []
    for moe_output in moe_outputs:
        if hasattr(moe_output, "tokens_per_expert"):
            tokens = moe_output.tokens_per_expert.float()
        elif hasattr(moe_output, "router_output"):
            tokens = moe_output.router_output.tokens_per_expert.float()
        else:
            continue

        if tokens.numel() == 0 or tokens.mean() == 0:
            continue

        mean_tokens.append(tokens.mean().item())
        load_imbalances.append((tokens.max() / tokens.mean()).item())
        threshold = tokens.sum() * 0.01
        active_expert_fracs.append((tokens > threshold).float().mean().item())

    if mean_tokens:
        metrics["mean_tokens_per_expert"] = sum(mean_tokens) / len(mean_tokens)
        metrics["mean_load_imbalance"] = sum(load_imbalances) / len(load_imbalances)
        metrics["mean_active_experts_frac"] = sum(active_expert_fracs) / len(active_expert_fracs)

    return metrics


def benchmark_resolution(experiment_name: str, model_spec: dict, resolution: int) -> dict:
    cfg = {**COMMON_MODEL_CONFIG, **model_spec["cfg"]}
    patch_size = cfg["patch_size"]
    row = {
        "experiment": experiment_name,
        "model": model_spec["model_name"],
        "label": model_spec["label"],
        "resolution": resolution,
        "batch_size": BATCH_SIZE,
        "time_window": TIME_WINDOW,
        "patch_size": patch_size,
        "tokens_per_sample": tokens_per_sample(resolution, patch_size),
        "pixels_per_batch": pixels_per_batch(resolution),
        "active_m": float("nan"),
        "total_m": float("nan"),
        "inference_ms": float("nan"),
        "inference_ms_std": float("nan"),
        "train_ms": float("nan"),
        "train_ms_std": float("nan"),
        "inference_vram_mb": float("nan"),
        "train_vram_mb": float("nan"),
        "inference_samples_per_s": float("nan"),
        "train_samples_per_s": float("nan"),
        "inference_pixels_per_s": float("nan"),
        "train_pixels_per_s": float("nan"),
        "mean_tokens_per_expert": float("nan"),
        "mean_load_imbalance": float("nan"),
        "mean_active_experts_frac": float("nan"),
        "status": "ok",
        "error": "",
    }

    torch._dynamo.reset()
    model = None
    try:
        batch = make_batch(resolution)
        model = get_model(model_spec["model_name"], **cfg).to(DEVICE)

        row["total_m"] = count_model_parameters(model, active=False) / 1e6
        row["active_m"] = count_model_parameters(model, active=True) / 1e6

        row["inference_ms"], row["inference_ms_std"] = measure_inference_ms(model, batch)
        row["train_ms"], row["train_ms_std"] = measure_train_step_ms(model, batch)
        row["inference_vram_mb"] = measure_inference_vram_mb(model, batch)
        row["train_vram_mb"] = measure_train_vram_mb(model, batch)

        inf_s = row["inference_ms"] / 1e3
        train_s = row["train_ms"] / 1e3
        row["inference_samples_per_s"] = BATCH_SIZE / inf_s
        row["train_samples_per_s"] = BATCH_SIZE / train_s
        row["inference_pixels_per_s"] = row["pixels_per_batch"] / inf_s
        row["train_pixels_per_s"] = row["pixels_per_batch"] / train_s

        if model_spec["is_moe"]:
            model.eval()
            with torch.no_grad():
                output = model(batch)
            row.update(extract_moe_metrics(output))

    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            row["status"] = "oom"
        else:
            row["status"] = "error"
        row["error"] = message.replace("\n", " ")[:500]
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    finally:
        if model is not None:
            del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return row


def write_csv(rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "problem_size_scaling.csv"
    fieldnames = [
        "experiment",
        "model",
        "label",
        "resolution",
        "batch_size",
        "time_window",
        "patch_size",
        "tokens_per_sample",
        "pixels_per_batch",
        "active_m",
        "total_m",
        "inference_ms",
        "inference_ms_std",
        "train_ms",
        "train_ms_std",
        "inference_vram_mb",
        "train_vram_mb",
        "inference_samples_per_s",
        "train_samples_per_s",
        "inference_pixels_per_s",
        "train_pixels_per_s",
        "mean_tokens_per_expert",
        "mean_load_imbalance",
        "mean_active_experts_frac",
        "status",
        "error",
    ]
    with out_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def print_row(row: dict) -> None:
    print(f"\n  [{row['label']}] @ {row['resolution']}x{row['resolution']}")
    print(f"  {'active':<18} {row['active_m']:>8.1f}M")
    print(f"  {'total':<18} {row['total_m']:>8.1f}M")
    print(f"  {'tokens/sample':<18} {row['tokens_per_sample']:>8}")
    print(f"  {'status':<18} {row['status']}")
    if row["status"] == "ok":
        print(f"  {'inference':<18} {row['inference_ms']:>8.1f} ± {row['inference_ms_std']:>5.1f} ms")
        print(f"  {'train step':<18} {row['train_ms']:>8.1f} ± {row['train_ms_std']:>5.1f} ms")
        print(f"  {'infer VRAM':<18} {row['inference_vram_mb']:>8.0f} MB")
        print(f"  {'train VRAM':<18} {row['train_vram_mb']:>8.0f} MB")
        print(f"  {'infer px/s':<18} {row['inference_pixels_per_s']:>8.0f}")
        print(f"  {'train px/s':<18} {row['train_pixels_per_s']:>8.0f}")
        if not torch.isnan(torch.tensor(row["mean_load_imbalance"])):
            print(f"  {'load imbalance':<18} {row['mean_load_imbalance']:>8.3f}")
            print(f"  {'active experts':<18} {row['mean_active_experts_frac']:>8.3f}")
    else:
        print(f"  {'error':<18} {row['error']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = (Path(args.out_dir) if args.out_dir else Path.home() / "temp") / "problem_size_report"

    print("=== Problem size scaling benchmark ===")
    print(
        f"Batch={BATCH_SIZE}, Time={TIME_WINDOW}, Channels={CHANNELS}, "
        f"Resolutions={RESOLUTIONS}, Patch={COMMON_MODEL_CONFIG['patch_size']}"
    )

    rows = []
    for experiment in EXPERIMENTS:
        print(f"\n{'=' * 72}")
        print(f"Experiment: {experiment['name']}")
        print(experiment["description"])
        print(f"{'=' * 72}")
        for resolution in RESOLUTIONS:
            print(f"\n{'-' * 72}")
            print(f"Resolution: {resolution}x{resolution}")
            print(f"{'-' * 72}")
            for model_spec in experiment["models"]:
                row = benchmark_resolution(experiment["name"], model_spec, resolution)
                rows.append(row)
                print_row(row)

    out_path = write_csv(rows, out_dir)
    print(f"\nWrote problem size scaling CSV to {out_path}")


if __name__ == "__main__":
    main()
