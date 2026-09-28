r"""
Convert BubbleML HDF5 files to webdataset shards.

Each HDF5 file contains time-series fields (dfun, temperature, velx, vely) with
shape [T, H, W], plus a sidecar JSON file with simulation parameters.

This script slices each file into non-overlapping windows of `--time_window_size`
timesteps and writes them as webdataset shards. Each sample in the shard contains:
  - <key>.npz  : (with --compress) compressed float32 array of shape [T, H, W, 4],
                 stored under the key "fields" (access as sample["npz"]["fields"])
  - <key>.npy  : (without --compress) uncompressed float32 array of shape [T, H, W, 4],
                 accessible directly as sample["npy"]
  - <key>.json : simulation parameters from the sidecar JSON

Samples are globally shuffled before writing so that temporally-adjacent chunks
from the same simulation are not co-located within shards.

Usage:
    python hdf5_dataset_to_webdataset.py \
        --hdf5_dir /data/BubbleML_2/PoolBoiling-* \
        --out_dir  /data/webdataset/poolboiling \
        --time_window_size 32 \
        --shard_size_gb 1
"""

import argparse
import glob
import io
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import webdataset as wds

FIELDS = ["dfun", "temperature", "velx", "vely"]


# All files not in val or test are used for training.
val_dataset = [
    "PoolBoiling-Subcooled-R515B-2D/Twall_47.hdf5",
    "PoolBoiling-Subcooled-FC72-2D/Twall_90.hdf5",
    "PoolBoiling-Saturated-R515B-2D/Twall_31.hdf5",
    "PoolBoiling-Saturated-LN2-2D/Twall_-175.hdf5",
    "PoolBoiling-Saturated-FC72-2D/Twall_100.hdf5",
]

test_dataset = [
    "PoolBoiling-Subcooled-R1233ZD-2D/Twall_60.hdf5",
    "PoolBoiling-Subcooled-OP250-2D/Twall_85.hdf5",
    "PoolBoiling-Subcooled-OP250-2D/Twall_97.hdf5",
    "PoolBoiling-Subcooled-LN2-2D/Twall_-170.hdf5",
    "PoolBoiling-Subcooled-LN2-2D/Twall_-155.hdf5",
    "PoolBoiling-Subcooled-LH2-2D/Twall_-209.hdf5",
    "PoolBoiling-Subcooled-LH2-2D/Twall_-225.hdf5",
    "PoolBoiling-Subcooled-FC72-2D/Twall_98.hdf5",
    "PoolBoiling-Subcooled-FC72-2D/Twall_110.hdf5",
    "PoolBoiling-Subcooled-R515B-2D/Twall_35.hdf5",
    "PoolBoiling-Subcooled-R515B-2D/Twall_13.hdf5",
    "PoolBoiling-Saturated-R515B-2D/Twall_33.hdf5",
    "PoolBoiling-Saturated-R515B-2D/Twall_9.hdf5",
    "PoolBoiling-Saturated-LN2-2D/Twall_-182.hdf5",
    "PoolBoiling-Saturated-LN2-2D/Twall_-166.hdf5",
    "PoolBoiling-Saturated-FC72-2D/Twall_86.hdf5",
    "PoolBoiling-Saturated-FC72-2D/Twall_107.hdf5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HDF5 dataset to webdataset shards.")
    parser.add_argument("--hdf5_dir", type=str, nargs="+", required=True,
                        help="One or more directories containing .hdf5 and matching .json files. "
                             "Shell globs are expanded by the shell, e.g. --hdf5_dir PoolBoiling-*")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for webdataset tar shards.")
    parser.add_argument("--compress", action="store_true", default=False)
    parser.add_argument("--time_window_size", type=int, default=32,
                        help="Number of timesteps per sample window.")
    parser.add_argument("--shard_size_gb", type=float, default=1.0,
                        help="Maximum size of each output tar shard in gigabytes.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for global shuffle.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="Number of worker processes for parallel chunk reading and compression.")
    return parser.parse_args()


def split_label(hdf5_path: str) -> str:
    """Return 'val', 'test', or 'train' for a given HDF5 file path."""
    parts = Path(hdf5_path)
    rel = f"{parts.parent.name}/{parts.name}"
    if any(rel.endswith(entry) for entry in val_dataset):
        return "val"
    if any(rel.endswith(entry) for entry in test_dataset):
        return "test"
    return "train"


def collect_chunk_indices(hdf5_path: str, time_window_size: int) -> list[tuple[str, int]]:
    """Return a list of (hdf5_path, chunk_idx) for every non-overlapping window."""
    with h5py.File(hdf5_path, "r") as h5file:
        num_timesteps = h5file[FIELDS[0]].shape[0]
    num_chunks = num_timesteps // time_window_size
    return [(hdf5_path, chunk_idx) for chunk_idx in range(num_chunks)]


def read_chunk(hdf5_path: str, chunk_idx: int, time_window_size: int) -> np.ndarray:
    """Read a single time-window chunk from an HDF5 file. Returns [T, H, W, 4] float32."""
    time_start = chunk_idx * time_window_size
    time_end = time_start + time_window_size
    with h5py.File(hdf5_path, "r") as h5file:
        return np.stack(
            [np.array(h5file[field][time_start:time_end], dtype=np.float32) for field in FIELDS],
            axis=-1,
        )


def process_chunk(args: tuple[str, int, int, dict]) -> tuple[str, bytes, bytes]:
    """Worker function: read one chunk, compress it, and return serialized bytes."""
    hdf5_path, compress, chunk_idx, time_window_size, sim_params = args
    stem = Path(hdf5_path).stem
    sample_key = f"{stem}_chunk_{chunk_idx:06d}"
    fields_array = read_chunk(hdf5_path, chunk_idx, time_window_size)
    array_key = "npz" if compress else "npy"
    return sample_key, array_key, array_to_bytes(fields_array, compress), json.dumps(sim_params).encode("utf-8")


def array_to_bytes(array: np.ndarray, compress: bool) -> bytes:
    buf = io.BytesIO()
    if compress:
        np.savez_compressed(buf, fields=array)
    else:
        np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def write_split(
    split_name: str,
    compress: bool,
    chunk_indices: list[tuple[str, int]],
    sim_params_cache: dict[str, dict],
    time_window_size: int,
    out_dir: str,
    max_shard_bytes: int,
    num_workers: int,
) -> int:
    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    shard_pattern = os.path.join(split_dir, "shard-%06d.tar")

    tasks = [
        (hdf5_path, compress, chunk_idx, time_window_size, sim_params_cache[hdf5_path])
        for hdf5_path, chunk_idx in chunk_indices
    ]

    total = 0
    with Pool(num_workers) as pool:
        results = pool.imap_unordered(process_chunk, tasks, chunksize=1)
        with wds.ShardWriter(shard_pattern, maxsize=max_shard_bytes) as sink:
            for sample_key, array_key, array_bytes, json_bytes in results:
                sink.write({"__key__": sample_key, array_key: array_bytes, "json": json_bytes})
                total += 1
    return total


def main():
    assert not (set(val_dataset) & set(test_dataset)), "val and test sets overlap"

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    hdf5_files = sorted(
        path
        for hdf5_dir in args.hdf5_dir
        for path in glob.glob(os.path.join(hdf5_dir, "**", "*.hdf5"), recursive=True)
    )
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under: {args.hdf5_dir}")

    # Load all sim params up front (small JSON files).
    sim_params_cache: dict[str, dict] = {}
    for hdf5_path in hdf5_files:
        json_path = hdf5_path.replace(".hdf5", ".json")
        with open(json_path, "r", encoding="utf-8") as f:
            sim_params_cache[hdf5_path] = json.load(f)

    # Collect all chunk indices per split, then globally shuffle each split.
    splits: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    for hdf5_path in hdf5_files:
        label = split_label(hdf5_path)
        chunks = collect_chunk_indices(hdf5_path, args.time_window_size)
        splits[label].extend(chunks)
        print(f"  [{label:5s}] {hdf5_path}  ({len(chunks)} chunks)")

    rng = random.Random(args.seed)
    for label in splits:
        rng.shuffle(splits[label])

    max_shard_bytes = int(args.shard_size_gb * 1024 ** 3)
    for split_name, chunk_indices in splits.items():
        if not chunk_indices:
            print(f"No files for split '{split_name}', skipping.")
            continue
        print(f"\nWriting {split_name} ({len(chunk_indices)} samples) ...")
        total = write_split(
            split_name=split_name,
            compress=args.compress,
            chunk_indices=chunk_indices,
            sim_params_cache=sim_params_cache,
            time_window_size=args.time_window_size,
            out_dir=args.out_dir,
            max_shard_bytes=max_shard_bytes,
            num_workers=args.num_workers,
        )
        print(f"  Done. {total} samples -> {os.path.join(args.out_dir, split_name)}/")


if __name__ == "__main__":
    main()
