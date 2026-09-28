"""
Benchmarks each stage of the WebDataset pipeline in isolation to identify
where time is spent: I/O, numpy decode, normalization, augmentation, or layout.

Usage:
    python bench/dataloader.py \
        --shards "/share/crsp/lab/amowli/share/BubbleML_2_wds_uncompressed/train/shard-{000000..000005}.tar" \
        --samples 20
"""
import argparse
import time
import random
import io
from contextlib import contextmanager
from collections import defaultdict

import braceexpand
import numpy as np
import torch
import webdataset as wds

from nucleus.data.forecast_webdataset import forecast_web_dataset


@contextmanager
def timer(label: str, totals: dict):
    t = time.perf_counter()
    yield
    totals[label] += time.perf_counter() - t


def benchmark(shard_urls: list[str], num_samples: int, history: int = 4, future: int = 4):
    totals = defaultdict(float)
    count = 0

    # Pass 1: raw I/O — time the tar iterator yielding each entry across all shards.
    print(f"Pass 1: measuring raw I/O time across {len(shard_urls)} shards...")
    io_times = []
    io_sizes = []
    raw_iter = iter(wds.WebDataset(shard_urls, shardshuffle=False))
    for _ in range(num_samples):
        t = time.perf_counter()
        try:
            sample = next(raw_iter)
            raw_bytes = sample.get("npz") or sample.get("npy") or b""
        except StopIteration:
            break
        io_times.append(time.perf_counter() - t)
        io_sizes.append(len(raw_bytes))

    n_io = len(io_times)
    io_total = sum(io_times)
    total_bytes = sum(io_sizes)
    gbps = (total_bytes / 1e9) / io_total
    print(f"  samples:     {n_io}")
    print(f"  per sample:  {io_total/n_io*1000:.1f} ms  |  {total_bytes/n_io/1e6:.1f} MB")
    print(f"  throughput:  {n_io/io_total:.2f} samples/sec  |  {gbps:.3f} GB/s\n")

    # Pass 2: per-stage processing breakdown.
    print("Pass 2: measuring per-stage processing time...")
    for sample in wds.WebDataset(shard_urls, shardshuffle=False):
        if count >= num_samples:
            break

        with timer("numpy_parse", totals):
            key = "npz" if "npz" in sample else "npy"
            raw_bytes = sample[key]
            if key == "npz":
                raw = np.load(io.BytesIO(raw_bytes))["fields"]
            else:
                raw = np.load(io.BytesIO(raw_bytes))

        with timer("to_tensor", totals):
            fields = torch.from_numpy(raw[:history + future].copy())
            inp = fields[:history]
            tgt = fields[history:]

        with timer("normalize", totals):
            mean = inp.mean(dim=(0, 1, 2), keepdim=True)
            std = inp.std(dim=(0, 1, 2), keepdim=True).clamp(min=1e-6)
            inp = (inp - mean) / std
            tgt = (tgt - mean) / std

        with timer("augment", totals):
            if random.random() < 0.5:
                inp = torch.flip(inp, dims=[2])
                tgt = torch.flip(tgt, dims=[2])

        with timer("layout_convert", totals):
            inp = inp.contiguous()
            tgt = tgt.contiguous()

        count += 1

    print(f"\nResults over {count} samples:")
    print(f"{'Stage':<20} {'Total (s)':>10} {'Per sample (ms)':>16} {'% of total':>10}")
    print("-" * 60)
    grand_total = sum(totals.values())
    for label, total in sorted(totals.items(), key=lambda x: -x[1]):
        per_sample_ms = total / count * 1000
        pct = total / grand_total * 100
        print(f"{label:<20} {total:>10.3f} {per_sample_ms:>16.1f} {pct:>10.1f}%")
    print("-" * 60)
    print(f"{'TOTAL':<20} {grand_total:>10.3f} {grand_total/count*1000:>16.1f} {'100.0%':>10}")
    print(f"\nEffective throughput: {count / grand_total:.2f} samples/sec")


def benchmark_parallel(shard_urls: list[str], num_samples: int, num_workers: int,
                       history: int = 4, future: int = 4):
    """Measures aggregate throughput of the full forecast_web_dataset pipeline."""
    from torch.utils.data import DataLoader

    dataset = forecast_web_dataset(
        shard_urls=shard_urls,
        cache_dir=None,
        cache_size=0,
        history_time_window=history,
        future_time_window=future,
        fluid_params=[],
        heater_params=[],
        global_params=[],
        layout="t h w c",
        normalizer=None,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=False)

    print(f"Parallel I/O: {num_workers} workers, {num_samples} samples...")
    total_bytes = 0
    count = 0
    t_start = time.perf_counter()
    for item in loader:
        # item is a Data namedtuple-like; measure bytes from the two main tensors
        total_bytes += item.input.nbytes + item.target.nbytes
        count += 1
        if count >= num_samples:
            break
    elapsed = time.perf_counter() - t_start

    gbps = (total_bytes / 1e9) / elapsed
    print(f"  samples:    {count}")
    print(f"  elapsed:    {elapsed:.2f}s")
    print(f"  throughput: {count/elapsed:.2f} samples/sec  |  {gbps:.3f} GB/s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards", required=True,
        help="Brace-expanded shard glob, e.g. 'path/shard-{000000..000005}.tar'"
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--future", type=int, default=4)
    parser.add_argument("--workers", type=int, default=None,
                        help="If set, run parallel I/O benchmark with this many workers instead")
    args = parser.parse_args()

    shard_urls = list(braceexpand.braceexpand(args.shards))
    print(f"Benchmarking {len(shard_urls)} shards, {args.samples} samples total\n")

    if args.workers is not None:
        benchmark_parallel(shard_urls, args.samples, args.workers, args.history, args.future)
    else:
        benchmark(shard_urls, args.samples, args.history, args.future)
