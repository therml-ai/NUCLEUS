import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import torch
from torch.utils.data import DataLoader
from boiling_data.flashx.batch import ConfigKeys
from boiling_data.tasks.forecast.flashx_batch import flashx_collater
from boiling_data.tasks.forecast.flashx_dataset import (
    FlashXForecastDataset,
    FlashXInMemoryForecastDataset,
)


SPLITS = ("train", "val", "test")
SPLIT_MANIFEST = "split.json"


def read_split(split_dir: Union[str, Path]) -> Dict[str, List[Path]]:
    """The cases of each split of a directory made by boiling-data's
    scripts/split_dataset.py: split_dir/<split>/<path> for every path its
    split.json lists under the split."""
    split_dir = Path(split_dir)
    with open(split_dir / SPLIT_MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)
    cases = {name: [split_dir / name / path for path in manifest[name]] for name in SPLITS}
    missing = [str(path) for paths in cases.values() for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{split_dir} is missing cases its {SPLIT_MANIFEST} lists: {missing}")
    return cases


@dataclass(frozen=True)
class ForecastDataSettings:
    input_timesteps: int
    target_timesteps: int
    batch_size: int
    start_frame: int = 0
    stride: int = 1
    unroll_steps: int = 1
    in_memory: bool = False
    num_workers: int = 8


def forecast_dataloader(
    paths: Sequence[Union[str, Path]],
    field_names: Sequence[str],
    config_keys: ConfigKeys,
    settings: ForecastDataSettings,
    shuffle: bool,
) -> DataLoader:
    """Batches of FlashXForecastBatch on the CPU, whose config tensors hold
    config_keys, the conditioning parameters a model reads."""
    dataset_class = FlashXInMemoryForecastDataset if settings.in_memory else FlashXForecastDataset
    dataset = dataset_class(
        list(paths),
        list(field_names),
        input_timesteps=settings.input_timesteps,
        target_timesteps=settings.target_timesteps,
        start_frame=settings.start_frame,
        stride=settings.stride,
        unroll_steps=settings.unroll_steps,
    )
    if len(dataset) == 0:
        raise ValueError(f"the files {list(paths)} hold no forecast windows of the requested size")
    workers = settings.num_workers
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        collate_fn=functools.partial(flashx_collater, config_keys=config_keys),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        # DataLoader rejects these without worker processes
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
