from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn
from boiling_data.bubbleml import read_bubbleml
from boiling_data.flashx.batch import FlashXBatch, batch_from_simulation, simulation_from_batch


@dataclass(frozen=True)
class RolloutSettings:
    start_frame: int
    history_timesteps: int
    trajectory_steps: int
    # the target window the model was trained on, at most history_timesteps
    target_timesteps: int
    # frames kept from each model call, at most target_timesteps
    frames_per_step: int


@torch.no_grad()
def rollout(
    model: nn.Module, history: FlashXBatch, num_steps: int, frames_per_step: int, target_timesteps: int
) -> FlashXBatch:
    """The num_steps frames a model predicts after a raw history, a few at a time,
    each call seeing the latest history.num_timesteps frames. The model's
    predict_next maps a raw window to the raw frames that follow it; it was trained
    to predict target_timesteps of them."""
    if not 1 <= frames_per_step <= target_timesteps <= history.num_timesteps:
        raise ValueError(
            f"need 1 <= frames_per_step <= target_timesteps <= {history.num_timesteps}, the history, not "
            f"frames_per_step={frames_per_step} and target_timesteps={target_timesteps}"
        )
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, not {num_steps}")
    window, predicted = history, None
    while predicted is None or predicted.num_timesteps < num_steps:
        remaining = num_steps - (0 if predicted is None else predicted.num_timesteps)
        frames = model.predict_next(window, min(frames_per_step, remaining), target_timesteps)
        predicted = frames if predicted is None else predicted.extend(frames)
        window = window.extend(frames).tail_time_window(history.num_timesteps)
    return predicted


def rollout_case(
    model: nn.Module, path: Union[str, Path], settings: RolloutSettings, device: torch.device
) -> tuple:
    """A rollout from the history frames of a BubbleML case, and the simulation time
    of every predicted frame, continuing the history's frame spacing."""
    history_frames = slice(settings.start_frame, settings.start_frame + settings.history_timesteps)
    simulation = read_bubbleml(path, field_names=model.field_names, frames=history_frames)
    if simulation.num_timesteps != settings.history_timesteps:
        raise ValueError(f"{path} has no {settings.history_timesteps} frames from frame {settings.start_frame}")
    history = batch_from_simulation(simulation, model.field_names, model.config_keys, device=device)
    predicted = rollout(
        model, history, settings.trajectory_steps, settings.frames_per_step, settings.target_timesteps
    )
    return predicted, _following_times(simulation.time, simulation.parameters.discretization, settings.trajectory_steps)


def save_rollout(prediction: FlashXBatch, time: np.ndarray, path: Union[str, Path]) -> Path:
    """Writes the first sample of a raw prediction as a BubbleML file, with the
    parameters and grid of its config."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return simulation_from_batch(prediction.to(torch.device("cpu")), time).to_bubbleml(path)


def _following_times(history_time: np.ndarray, discretization: dict, num_steps: int) -> np.ndarray:
    if len(history_time) > 1:
        step = (history_time[-1] - history_time[0]) / (len(history_time) - 1)
    else:
        step = float(discretization["dt"])
    return history_time[-1] + step * np.arange(1, num_steps + 1)
