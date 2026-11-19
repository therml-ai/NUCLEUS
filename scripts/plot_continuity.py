"""Checks a rollout against continuity, div(u) = S, where S is the divergence that
phase change at the interface gives, computed from the rollout's own temperature and
sdf as Flash-X does, next to the same checks on the case it was rolled out from.

The figure's top row maps the residual div(u) - S on one frame and histograms it
near and away from the interface. The bottom row follows the whole rollout: the
vapor volume the sdf holds against the volume the evaporation predicts, the gap
between them, and the net outflow through the top against the integral of S.

    python scripts/plot_continuity.py rollouts/case.hdf5 split/test/case.hdf5
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from boiling_data.boiling_data import BoilingSimulation
from boiling_data.bubbleml import read_bubbleml
from boiling_data.flashx.batch import batch_from_simulation
from matplotlib.axes import Axes
from matplotlib.colors import SymLogNorm

from nucleus.models.nucleus2_moe_divfree_projection import DivFreePhysics, continuity_source, divfree_physics
from nucleus.physics.poisson import divergence_centers_from_faces

FIELDS = ("dfun", "temperature", "velfacex", "velfacey")
# frames per continuity call, which bounds the memory of its extrapolation
CHUNK_FRAMES = 16
# the residual is mapped over this many decades below its largest magnitude
RESIDUAL_DECADES = 6
# cells within this many cells of the interface count as the interface band
BAND_CELLS = 3
# the smoothed Heaviside's half width in cells, so the vapor volume does not jump
# as the interface crosses cell centers
HEAVISIDE_CELLS = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollout", type=Path, help="a rollout written by scripts/rollout.py")
    parser.add_argument("source", type=Path, help="the BubbleML case the rollout started from")
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="frame index whose residual is mapped in both files; the last plotted by default",
    )
    parser.add_argument(
        "--num-frames", type=int, default=250, help="frames plotted from the start of each file, at most"
    )
    parser.add_argument("--output", type=Path, help="PNG to write; <rollout name>_continuity.png by default")
    return parser.parse_args()


@dataclass
class Continuity:
    """One case's continuity diagnostics over its frames: the residual div(u) - S
    with the top outflow row zeroed, where the outflow condition breaks div = S, and
    the per-frame relative residual |div(u) - S| / |S| (L2 over the frame), and the
    per-frame integrals of S, the vapor volume, the vapor leaving through the top,
    and the net outflow through the top."""

    time: np.ndarray
    residual: np.ndarray
    relative_residual: np.ndarray
    near_interface: np.ndarray
    source_integral: np.ndarray
    vapor_volume: np.ndarray
    vapor_outflow: np.ndarray
    top_outflow: np.ndarray
    rhogas: float


def smoothed_heaviside(sdf: torch.Tensor, half_width: float) -> torch.Tensor:
    """1 in the vapor (sdf > 0), 0 in the liquid, with a smooth step across the interface."""
    scaled = (sdf / half_width).clamp(-1.0, 1.0)
    return 0.5 * (1.0 + scaled + torch.sin(torch.pi * scaled) / torch.pi)


def chunk_diagnostics(fields: dict[str, torch.Tensor], physics: DivFreePhysics) -> dict[str, torch.Tensor]:
    """The diagnostics of a batch of one sample's fields, each [1, T, ...]."""
    dx, dy = physics.grid.dx, physics.grid.dy
    source = continuity_source(fields["temperature"], fields["dfun"], physics)
    divergence = divergence_centers_from_faces(fields["velfacex"], fields["velfacey"], dx, dy)
    residual = divergence - source
    residual[..., -1, :] = 0.0
    source_norm = source[..., :-1, :].norm(dim=(-2, -1)).clamp_min(1e-30)
    vapor = smoothed_heaviside(fields["dfun"], HEAVISIDE_CELLS * dx)
    top_velocity = fields["velfacey"][..., -1, :]
    return {
        "residual": residual[0],
        "relative_residual": residual[0].norm(dim=(-2, -1)) / source_norm[0],
        "near_interface": (fields["dfun"].abs() < BAND_CELLS * dx)[0],
        "source_integral": source[0].sum(dim=(-2, -1)) * dx * dy,
        "vapor_volume": vapor[0].sum(dim=(-2, -1)) * dx * dy,
        "vapor_outflow": (vapor[0, ..., -1, :] * top_velocity[0]).sum(dim=-1) * dx,
        "top_outflow": top_velocity[0].sum(dim=-1) * dx,
    }


def continuity_diagnostics(simulation: BoilingSimulation, frames: np.ndarray) -> Continuity:
    batch = batch_from_simulation(simulation, FIELDS, dtype=torch.float64)
    physics = divfree_physics(batch)
    chunks = []
    for start in range(0, len(frames), CHUNK_FRAMES):
        chunk_frames = torch.from_numpy(frames[start:start + CHUNK_FRAMES])
        fields = {name: tensor[:, chunk_frames] for name, tensor in batch.fields.items()}
        chunks.append(chunk_diagnostics(fields, physics))
    joined = {name: torch.cat([chunk[name] for chunk in chunks]).numpy() for name in chunks[0]}
    return Continuity(
        time=simulation.time[frames],
        rhogas=float(simulation.parameters.non_dimensional["rhogas"]),
        **joined,
    )


def cumulative_integral(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    """The trapezoidal integral of values over time from the first frame."""
    steps = 0.5 * (values[1:] + values[:-1]) * np.diff(time)
    return np.concatenate([[0.0], np.cumsum(steps)])


def predicted_vapor_volume(continuity: Continuity) -> np.ndarray:
    """The vapor volume evaporation and the outflow predict from the first frame's.
    Evaporating liquid of volume dV becomes vapor of volume dV / rhogas, so the
    vapor grows by the integral of S over (1 - rhogas), less the vapor that leaves
    through the top."""
    growth_rate = continuity.source_integral / (1.0 - continuity.rhogas) - continuity.vapor_outflow
    return continuity.vapor_volume[0] + cumulative_integral(growth_rate, continuity.time)


def plot_residual_map(axes: Axes, simulation: BoilingSimulation, residual: np.ndarray, norm, title: str) -> None:
    dfun = simulation.field("dfun")
    image = axes.pcolormesh(dfun.grid_x, dfun.grid_y, residual, shading="nearest", cmap="RdBu_r", norm=norm)
    axes.set_title(title, fontsize=10)
    axes.set_aspect("equal")
    axes.figure.colorbar(image, ax=axes, fraction=0.05)


def plot_residual_histogram(axes: Axes, cases: dict[str, Continuity], frame: int) -> None:
    """|div(u) - S| over every frame, near and away from the interface."""
    largest = max(np.abs(case.residual).max() for case in cases.values()) or 1.0
    bins = np.logspace(np.log10(largest) - 12, np.log10(largest), 60)
    for (name, case), color in zip(cases.items(), ("black", "tab:red")):
        magnitude = np.abs(case.residual)
        for mask, style, where in ((case.near_interface, "-", "near"), (~case.near_interface, "--", "away")):
            values = np.clip(magnitude[mask], bins[0], None)
            axes.hist(values, bins=bins, histtype="step", color=color, linestyle=style, label=f"{name}, {where}")
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("|div(u) - S|")
    axes.set_title(f"residual over every frame,\nnear (< {BAND_CELLS} cells) and away from the interface", fontsize=10)
    axes.legend(fontsize=8)


def plot_time_series(axes: np.ndarray, cases: dict[str, Continuity]) -> None:
    for (name, case), color in zip(cases.items(), ("black", "tab:red")):
        predicted = predicted_vapor_volume(case)
        index = np.arange(len(case.time))
        axes[0].plot(index, case.vapor_volume, color=color, label=f"{name}, from the sdf")
        axes[0].plot(index, predicted, color=color, linestyle="--", label=f"{name}, from S")
        axes[1].plot(index, (case.vapor_volume - predicted) / predicted.mean(), color=color, label=name)
        axes[2].plot(index, case.top_outflow, color=color, label=f"{name}, top outflow")
        axes[2].plot(index, case.source_integral, color=color, linestyle="--", label=f"{name}, integral of S")
    titles = (
        "vapor volume",
        "vapor volume gap, (sdf - S prediction) / mean volume",
        "net outflow through the top vs integral of S",
    )
    for panel, title in zip(axes, titles):
        panel.set_title(title, fontsize=10)
        panel.set_xlabel("frame")
        panel.legend(fontsize=8)


def plot_continuity(
    rollout: BoilingSimulation, source: BoilingSimulation, num_frames: int, frame: int, title: str
):
    """Each file's first num_frames frames, against frame index. frame: the frame
    index mapped in both, wrapped into the frames each file has."""
    cases = {
        "Flash-X": continuity_diagnostics(source, np.arange(min(num_frames, source.num_timesteps))),
        "rollout": continuity_diagnostics(rollout, np.arange(min(num_frames, rollout.num_timesteps))),
    }
    mapped = {name: frame % len(case.time) for name, case in cases.items()}
    largest = max(np.abs(case.residual[mapped[name]]).max() for name, case in cases.items()) or 1.0
    norm = SymLogNorm(linthresh=largest / 10**RESIDUAL_DECADES, vmin=-largest, vmax=largest)
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for panel, (name, case), simulation in zip(axes[0], cases.items(), (source, rollout)):
        index = mapped[name]
        map_title = (
            f"{name}, frame {index}: div(u) - S\n"
            f"|div(u) - S| / |S| = {case.relative_residual[index]:.2g}"
        )
        plot_residual_map(panel, simulation, case.residual[index], norm, map_title)
    plot_residual_histogram(axes[0, 2], cases, frame)
    plot_time_series(axes[1], cases)
    figure.suptitle(title)
    return figure


def main() -> None:
    args = parse_args()
    rollout = read_bubbleml(args.rollout, field_names=FIELDS)
    source = read_bubbleml(args.source, field_names=FIELDS)
    title = f"continuity of {args.rollout.name} against {args.source.name}"
    figure = plot_continuity(rollout, source, args.num_frames, args.frame, title)
    output = args.output or args.rollout.with_name(f"{args.rollout.stem}_continuity.png")
    figure.savefig(output, dpi=130)
    plt.close(figure)
    print(f"-> {output}")


if __name__ == "__main__":
    main()
