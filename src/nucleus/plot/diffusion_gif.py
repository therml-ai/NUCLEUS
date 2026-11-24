from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from nucleus.data.layout import convert_layout


def save_rollout_gif(
    frame_paths: list[Path], output_path: Path,
    max_frames: int = 120, fps: int = 10,
) -> Path:
    if max_frames < 1 or fps < 1:
        raise ValueError("GIF max_frames and fps must be positive.")
    if not frame_paths:
        raise ValueError("At least one rollout frame is required.")
    sample_indices = np.linspace(
        0, len(frame_paths) - 1, min(max_frames, len(frame_paths)), dtype=int,
    )
    frames = []
    for frame_index in sample_indices:
        with Image.open(frame_paths[frame_index]) as image:
            frames.append(image.convert("RGB"))
    canvas_size = (
        max(frame.width for frame in frames),
        max(frame.height for frame in frames),
    )
    padded_frames = []
    for frame in frames:
        canvas = Image.new("RGB", canvas_size, "white")
        canvas.paste(frame, (0, 0))
        padded_frames.append(canvas)
    padded_frames[0].save(
        output_path, save_all=True, append_images=padded_frames[1:],
        duration=max(1, round(1000 / fps)), loop=0,
    )
    return output_path


def _render_sdf_frame(fields: dict, frame_index: int, color_limit: float) -> Image.Image:
    figure = Figure(figsize=(12, 3), dpi=90)
    canvas = FigureCanvasAgg(figure)
    for panel_index, (title, field) in enumerate(fields.items(), start=1):
        axis = figure.add_subplot(1, len(fields), panel_index)
        axis.imshow(field, cmap="RdBu_r", vmin=-color_limit, vmax=color_limit)
        if field.min() < 0 < field.max():
            axis.contour(field, levels=[0], colors="black", linewidths=0.5)
        axis.set_title(title, fontsize=9)
        axis.set_axis_off()
    figure.suptitle(f"Rollout frame {frame_index} | shared SDF scale ±{color_limit:.3g}")
    figure.tight_layout()
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy())


def _window_fields(window: dict, layout: str, sdf_index: int) -> dict:
    fields = {
        "Ground truth": convert_layout(window["targets"], "t h w c", layout)[..., sdf_index],
        "Surrogate": window["surrogate_sdf"],
        "Surrogate reinitialized": window["surrogate_reinitialized_sdf"],
        "Diffusion raw": window["diffusion_raw_sdf"],
        "Diffusion reinitialized": convert_layout(window["preds"], "t h w c", layout)[..., sdf_index],
    }
    fields.update({
        name: values for name, values in window.items()
        if name.startswith("diffusion_gaussian_")
    })
    return fields


def save_diffusion_gif(
    window_paths: list[Path], layout: str, sdf_index: int,
    output_path: Path, max_frames: int = 120, fps: int = 10,
    variant: str | None = None,
) -> Path:
    if max_frames < 1 or fps < 1:
        raise ValueError("GIF max_frames and fps must be positive.")
    if not window_paths:
        raise ValueError("At least one rollout window is required.")
    selected_frames = []
    total_frames = 0
    for window_path in window_paths:
        window = torch.load(window_path, map_location="cpu", weights_only=False)
        frame_count = window["diffusion_raw_sdf"].shape[0]
        selected_frames.extend((window_path, offset, total_frames + offset) for offset in range(frame_count))
        total_frames += frame_count
    sample_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    samples = [selected_frames[index] for index in sample_indices]
    color_limit = _shared_color_limit(samples, layout, sdf_index)
    frames = _render_samples(samples, layout, sdf_index, color_limit, variant)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=max(1, round(1000 / fps)), loop=0)
    return output_path


def _shared_color_limit(samples: list, layout: str, sdf_index: int) -> float:
    color_limit = 0.0
    cached_path = None
    for window_path, offset, _ in samples:
        if window_path != cached_path:
            window = torch.load(window_path, map_location="cpu", weights_only=False)
            fields = _window_fields(window, layout, sdf_index)
            cached_path = window_path
        color_limit = max(color_limit, *(field[offset].abs().max().item() for field in fields.values()))
    return max(color_limit, 1e-6)


def _render_samples(
    samples: list, layout: str, sdf_index: int, color_limit: float,
    variant: str | None = None,
) -> list:
    frames = []
    cached_path = None
    for window_path, offset, frame_index in samples:
        if window_path != cached_path:
            window = torch.load(window_path, map_location="cpu", weights_only=False)
            fields = _window_fields(window, layout, sdf_index)
            if variant is not None:
                fields = {
                    "Ground truth": fields["Ground truth"],
                    "Original diffusion": fields["Diffusion reinitialized"],
                    f"Gaussian σ={variant.rsplit('_', 2)[1]}.{variant.rsplit('_', 1)[1]} cells": fields[variant],
                }
            else:
                fields = {title: field for title, field in fields.items()
                          if not title.startswith("diffusion_gaussian_")}
            cached_path = window_path
        frame_fields = {title: field[offset].numpy() for title, field in fields.items()}
        frames.append(_render_sdf_frame(frame_fields, frame_index, color_limit))
    return frames
