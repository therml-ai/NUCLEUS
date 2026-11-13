import math
import gc
import os
from datetime import date

import torch
import wandb
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import hydra
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from nucleus.data import InMemForecastDataset
from nucleus.data.batching import collate
from nucleus.data.layout import convert_layout
from nucleus.data.normalize import get_normalizer
from nucleus.models import load_model_from_checkpoint
from nucleus.models.nucleus2_moe_divfree import (
    Nucleus2MoEDivFreeInput,
    cells_to_x_face,
    cells_to_y_face,
    domain_x_coords,
)
from nucleus.utils.set_fp32_precision import set_fp32_precision
from nucleus.utils.sdf_reinit import sdf_reinit_sussman
from nucleus.utils.sdf_metrics import sdf_frame_metrics
from nucleus.utils.sdf_smoothing import smooth_sdf_frames
from nucleus.utils.physical_metrics import PhysicalMetrics, BubbleMetrics, physical_metrics, bubble_metrics
from nucleus.test import TestResults
from nucleus.plot.plotting import plot_rollout
from nucleus.plot.diffusion_gif import save_diffusion_gif, save_rollout_gif
from einops import rearrange
from pathlib import Path


GAUSSIAN_SIGMA_CELLS = (0.5, 1.0)

def training_loop()


if __name__ == "__main__":
    main()
