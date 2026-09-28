from typing import List, Optional, Union
import random

import braceexpand
import torch
import webdataset as wds

from nucleus.data.batching import make_data, Data
from nucleus.data.layout import convert_layout
from nucleus.data.normalize import Normalizer

def forecast_web_dataset(
    shard_urls: Union[str, List[str]],
    cache_dir: str,
    cache_size: int,
    history_time_window: int,
    future_time_window: int,
    fluid_params: List[str],
    heater_params: List[str],
    global_params: List[str],
    layout: str,
    normalizer: Optional[Normalizer],
    augment: bool,
    shuffle_buffer: int = 32,
):
    def _decode_sample(sample: dict) -> Data:
        raw = sample["npz"]["fields"] if "npz" in sample else sample["npy"]
        fields = torch.from_numpy(raw[:history_time_window + future_time_window].copy())  # [T, H, W, 4]
        sim_params = sample["json"]

        inp = fields[:history_time_window]
        tgt = fields[history_time_window:]

        bulk_temp = int(sim_params["bulk_temp"])
        if normalizer is not None:
            inp = normalizer.normalize(inp, bulk_temp)
            tgt = normalizer.normalize(tgt, bulk_temp)
            sim_params = normalizer.normalize_params([sim_params])[0]

        if augment and random.random() < 0.5:
            inp = torch.flip(inp, dims=[2])
            tgt = torch.flip(tgt, dims=[2])

        inp = convert_layout(inp, layout)
        tgt = convert_layout(tgt, layout)

        return make_data(
            input=inp.float(),
            target=tgt.float(),
            sim_params_dict=sim_params,
            downsample_factor=1,
            fluid_params=fluid_params,
            heater_params=heater_params,
            global_params=global_params
        )
        
    def _decode_samples(samples):
        for sample in samples:
            yield _decode_sample(sample)
    
    if isinstance(shard_urls, str):
        shard_urls = list(braceexpand.braceexpand(shard_urls))

    pipeline = (
        wds.WebDataset(
            shard_urls,
            cache_dir=cache_dir,
            cache_size=cache_size,
            shardshuffle=100 if augment else 0,
            # Partition shards across nodes and workers so each worker reads a
            # disjoint subset. Without this, all 8 workers read all shards,
            # causing 8x redundant NFS I/O and duplicate samples per epoch.
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )
        .decode()
    )
    if augment:
        pipeline = pipeline.shuffle(shuffle_buffer)
    pipeline.append(_decode_samples)
    return pipeline