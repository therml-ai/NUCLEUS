import io
import json
import tempfile
import os
from pathlib import Path

import webdataset as wds
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from nucleus.data.forecast_webdataset import forecast_web_dataset
from nucleus.data.batching import collate

FIELDS = ["dfun", "temperature", "velx", "vely"]

NUM_TIMESTEPS = 64
HEIGHT = 32
WIDTH = 32
HISTORY = 8
FUTURE = 4


DUMMY_SIM_PARAMS = {
    "bulk_temp": 60.0,
    "sat_temp": 56.0,
    "inv_reynolds": 0.01,
    "cpgas": 1.0,
    "mugas": 1e-5,
    "rhogas": 1.2,
    "thcogas": 0.025,
    "stefan": 0.5,
    "prandtl": 0.7,
    "gravy": -9.8,
    "x_min": 0.0,
    "x_max": 1.0,
    "y_min": 0.0,
    "y_max": 1.0,
    "num_blocks_x": 4,
    "nx_block": 8,
    "num_blocks_y": 4,
    "ny_block": 8,
    "heater": {
        "wallTemp": 76.0,
        "nucWaitTime": 0.1,
        "rcdAngle": 30.0,
        "advAngle": 45.0,
        "velContact": 0.01,
        "xMin": 0.2,
        "xMax": 0.8,
    },
}


def write_dummy_shard(directory: str, num_chunks: int = 4) -> str:
    """Write a single webdataset tar shard with `num_chunks` dummy samples."""
    shard_path = os.path.join(directory, "shard-000000.tar")
    rng = np.random.default_rng(0)

    with wds.TarWriter(shard_path) as sink:
        for chunk_idx in range(num_chunks):
            fields = rng.random((NUM_TIMESTEPS, HEIGHT, WIDTH, 4), dtype=np.float32)

            buf = io.BytesIO()
            np.savez_compressed(buf, fields=fields)
            npz_bytes = buf.getvalue()

            sink.write({
                "__key__": f"dummy_chunk_{chunk_idx:06d}",
                "npz": npz_bytes,
                "json": json.dumps(DUMMY_SIM_PARAMS).encode("utf-8"),
            })

    return shard_path


@pytest.fixture
def shard_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield write_dummy_shard(tmpdir, num_chunks=4)


def make_dataset(shard_path: str, augment: bool = False) -> wds.WebDataset:
    return forecast_web_dataset(
        shard_urls=[shard_path],
        cache_dir=None,
        cache_size=-1,
        history_time_window=HISTORY,
        future_time_window=FUTURE,
        fluid_params=["inv_reynolds", "cpgas", "mugas", "rhogas", "thcogas", "stefan", "prandtl", "gravy", "bulk_temp"],
        heater_params=["wallTemp", "xMin", "xMax"],
        global_params=["gravy"],
        layout="t h w c",
        normalizer=None,
        augment=augment,
    )


def test_dataset_yields_samples(shard_path):
    dataset = make_dataset(shard_path)
    samples = list(dataset)
    assert len(samples) == 4


def test_input_shape(shard_path):
    sample = next(iter(make_dataset(shard_path)))
    assert sample.input.shape == (HISTORY, HEIGHT, WIDTH, 4)


def test_target_shape(shard_path):
    sample = next(iter(make_dataset(shard_path)))
    assert sample.target.shape == (FUTURE, HEIGHT, WIDTH, 4)


def test_output_dtype_is_float32(shard_path):
    sample = next(iter(make_dataset(shard_path)))
    assert sample.input.dtype == torch.float32
    assert sample.target.dtype == torch.float32


def test_sim_params_tensor_shape(shard_path):
    sample = next(iter(make_dataset(shard_path)))
    # fluid(9) + heater(3) + global(1) = 13
    assert sample.sim_params_tensor is not None
    assert sample.sim_params_tensor.shape == (13,)


def test_dataloader_batching(shard_path):
    dataset = make_dataset(shard_path)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    assert batch.input.shape == (2, HISTORY, HEIGHT, WIDTH, 4)
    assert batch.target.shape == (2, FUTURE, HEIGHT, WIDTH, 4)


def test_input_and_target_are_disjoint(shard_path):
    """The target should be the timesteps immediately after the input, not overlapping."""
    dataset = make_dataset(shard_path)
    sample = next(iter(dataset))
    # input uses timesteps [0, HISTORY), target uses [HISTORY, HISTORY+FUTURE)
    # They should not be identical (different time slices of the same chunk)
    assert not torch.equal(sample.input[:FUTURE], sample.target)
