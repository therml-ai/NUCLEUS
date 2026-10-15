import diffusion
import os
import math
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, Dict, List, Union
from torch.utils.data import DataLoader

from nucleus.models import get_model
from nucleus.data.batching import CollatedBatch, collate
from nucleus.data.normalize import get_normalizer
from nucleus.data import InMemForecastDataset
from nucleus.test import run_test, TestResults
from nucleus.plot.plotting import (
    plot_rollout,
    plot_distribution,
)
from nucleus.utils.set_fp32_precision import set_fp32_precision
from nucleus.utils.sdf_reinit import sdf_reinit_sussman

import hydra
from omegaconf import DictConfig, OmegaConf
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from lightning import LightningModule

def build_diffusion_model(cfg: DictConfig, device: torch.device):
    model_name = cfg.model_cfg.name
    model_kwargs = {
        "input_fields": 4,
        "output_fields": 4,
        "patch_size": cfg.model_cfg.params.patch_size,
        "embed_dim": cfg.model_cfg.params.embed_dim,
        "processor_blocks": cfg.model_cfg.params.processor_blocks,
        "num_heads": cfg.model_cfg.params.num_heads,
        "num_fluid_params": cfg.model_cfg.params.num_fluid_params,
    }
    if cfg.model_cfg.params.get("num_experts", None) is not None:
        model_kwargs["num_experts"] = cfg.model_cfg.params.num_experts
        model_kwargs["topk"] = cfg.model_cfg.params.topk

    backbone = get_model(model_name, **model_kwargs).to(device)
    backbone.eval()

    if cfg.get("checkpoint_path") and os.path.isfile(cfg.checkpoint_path):
        load_backbone_weights(backbone, cfg.checkpoint_path, device)

    model = DiffusionModelWrapper(
        backbone=backbone,
        patch_size=cfg.model_cfg.params.patch_size,
        embed_dim=cfg.model_cfg.params.embed_dim,
    ).to(device)
    return model


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    set_fp32_precision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action = cfg.get("action", "train") 
    if action == "train":
        train_diffusion(cfg)
        return

    if action == "inf":
        model = build_diffusion_model(cfg, device)
        ckpt_path = cfg.get("checkpoint_path")
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            scheduler_config = ckpt.get("noise_scheduler_config", None)
        else:
            scheduler_config = None

        num_train_timesteps = cfg.get("num_train_timesteps", 1000)
        num_inference_steps = cfg.get("num_inference_steps", 100)
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=cfg.get("beta_start", 1e-4),
            beta_end=cfg.get("beta_end", 0.02),
            prediction_type="epsilon",
        )
        if scheduler_config is not None:
            noise_scheduler = DDPMScheduler.from_config(scheduler_config)

        noise_scheduler.set_timesteps(num_inference_steps)
        normalizer = get_normalizer(
            OmegaConf.to_container(cfg.normalizer_cfg, resolve=True)
        )

        save_root = pathlib.Path(cfg.checkpoint_path).parent / "diffusion_rollouts"
        save_root.mkdir(parents=True, exist_ok=True)
        all_test_results = []
        for test_file_path in cfg.data_cfg.test_paths:
            test_results = run_diffusion_test(
                cfg, model, noise_scheduler, normalizer,
                test_file_path, max_timesteps=1000,
                num_inference_steps=num_inference_steps, device=device,
            )
            all_test_results.append(test_results)

            save_dir = save_root / f"{test_results.case_name}"
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_rollout(
                save_dir=save_dir,
                rollout=test_results.preds,
                test_results=test_results,
                step_size=5,
                include_ground_truth=True,
            )
            plot_distribution(
                save_dir=save_dir,
                rollout=test_results.preds,
                test_results=test_results,
            )

        torch.save(all_test_results, save_root / "diffusion_test_results.pt")
        return

    # Default: train + evaluate on validation
    model = train_diffusion(cfg)
    model.eval()

    normalizer = get_normalizer(
        OmegaConf.to_container(cfg.normalizer_cfg, resolve=True)
    )
    num_inference_steps = cfg.get("num_inference_steps", 100)
    num_train_timesteps = cfg.get("num_train_timesteps", 1000)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=cfg.get("beta_start", 1e-4),
        beta_end=cfg.get("beta_end", 0.02),
        prediction_type="epsilon",
    )
    noise_scheduler.set_timesteps(num_inference_steps)

    save_root = pathlib.Path(cfg.get("output_dir", "diffusion_output")) / "rollouts"
    save_root.mkdir(parents=True, exist_ok=True)
    all_test_results = []
    for test_file_path in cfg.data_cfg.test_paths:
        test_results = run_diffusion_test(
            cfg, model, noise_scheduler, normalizer,
            test_file_path, max_timesteps=1000,
            num_inference_steps=num_inference_steps, device=device,
        )
        all_test_results.append(test_results)
        save_dir = save_root / f"{test_results.case_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_rollout(
            save_dir=save_dir,
            rollout=test_results.preds,
            test_results=test_results,
            step_size=5,
            include_ground_truth=True,
        )
        plot_distribution(
            save_dir=save_dir,
            rollout=test_results.preds,
            test_results=test_results,
        )

    torch.save(all_test_results, save_root / "diffusion_test_results.pt")


if __name__ == "__main__":
    main()
