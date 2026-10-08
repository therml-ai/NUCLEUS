from typing import Tuple

import braceexpand
import lightning as L
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from nucleus.data import get_pydataset, forecast_web_dataset
from nucleus.data.normalize import get_normalizer
from nucleus.flashx.divfree import FlashXDivFreeDataset, flashx_divfree_collate
from nucleus.models.modules import get_train_module

WEB_PYDATASET = "forecast_web"
# datasets reading BubbleML files written by boiling-data, kept apart from the
# nucleus.data registry so they share no code with the older datasets
FLASHX_PYDATASETS = {
    "flashx_divfree": (FlashXDivFreeDataset, flashx_divfree_collate),
}


def dataset_and_collate(pydataset: str):
    if pydataset in FLASHX_PYDATASETS:
        return FLASHX_PYDATASETS[pydataset]
    return get_pydataset(pydataset)


def build_train_module(cfg: DictConfig) -> L.LightningModule:
    return get_train_module(cfg.model_cfg.train_module_name)(
        checkpoint_path=cfg.checkpoint_path,
        model_cfg=cfg.model_cfg,
        data_cfg=cfg.data_cfg,
        normalizer_cfg=cfg.normalizer_cfg,
        optim_cfg=cfg.optim_cfg,
        scheduler_cfg=cfg.scheduler_cfg,
        log_wandb=False,
    )


def build_datasets(cfg: DictConfig, train_module: L.LightningModule) -> Tuple[Dataset, Dataset]:
    model = train_module.model
    shared_kwargs = dict(
        history_time_window=cfg.history_time_window,
        future_time_window=cfg.future_time_window,
        fluid_params=model.expected_fluid_params,
        heater_params=model.expected_heater_params,
        global_params=model.expected_global_params,
        layout=model.layout,
        normalizer=get_normalizer(OmegaConf.to_container(cfg.normalizer_cfg, resolve=True)),
    )
    if cfg.pydataset == WEB_PYDATASET:
        return _build_web_datasets(cfg, shared_kwargs)
    dataset_cls, _ = dataset_and_collate(cfg.pydataset)
    hdf5_kwargs = dict(
        time_step=cfg.time_step,
        start_time=cfg.start_time,
        input_fields=cfg.data_cfg.input_fields,
        output_fields=cfg.data_cfg.output_fields,
    )
    train_dataset = dataset_cls(filenames=cfg.data_cfg.train_paths, augment=True, **shared_kwargs, **hdf5_kwargs)
    val_dataset = dataset_cls(filenames=cfg.data_cfg.val_paths, augment=False, **shared_kwargs, **hdf5_kwargs)
    return train_dataset, val_dataset


def _build_web_datasets(cfg: DictConfig, shared_kwargs: dict) -> Tuple[Dataset, Dataset]:
    train_shard_urls = list(braceexpand.braceexpand(list(cfg.data_cfg.train_paths)[0]))
    web_kwargs = dict(cache_dir=None, cache_size=0, **shared_kwargs)
    train_dataset = forecast_web_dataset(shard_urls=train_shard_urls, augment=True, **web_kwargs)
    val_dataset = forecast_web_dataset(shard_urls=list(cfg.data_cfg.val_paths)[0], augment=False, **web_kwargs)
    return train_dataset, val_dataset


def build_dataloaders(
    cfg: DictConfig,
    train_module: L.LightningModule,
    train_workers: int = 8,
    val_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = build_datasets(cfg, train_module)
    _, collate_fn = dataset_and_collate(cfg.pydataset)
    is_web = cfg.pydataset == WEB_PYDATASET
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=not is_web,
        collate_fn=collate_fn,
        **_worker_kwargs(train_workers, prefetch_factor=2, persistent=not is_web),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **_worker_kwargs(val_workers, prefetch_factor=3, persistent=not is_web),
    )
    return train_dataloader, val_dataloader


def _worker_kwargs(num_workers: int, prefetch_factor: int, persistent: bool) -> dict:
    # DataLoader rejects prefetch_factor and persistent_workers without worker
    # processes, which smoke runs use so errors surface in the main process
    if num_workers == 0:
        return dict(num_workers=0, pin_memory=True)
    return dict(
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent,
        multiprocessing_context="fork",
    )
