"""Visualize the dynamics of the Helmholtz potentials psi/phi on real data.

Loads a trajectory's face velocities from an HDF5 file, computes the nodal
streamfunction ``psi`` and cell-centered potential ``phi`` with
``helmholtz_from_faces``, and answers two questions the velocity-loss spike
raised:

  1. Do psi/phi move continuously across timesteps compared with the velocity?
     (per-step increment RMS, normalized per field, over time)
  2. Where does each field's spatial variance live -- large scales (potentials)
     vs small scales (velocity)? (radially averaged power spectrum)

It also prints the solver round-trip floor: reconstructing the velocity from the
computed psi/phi and comparing to the stored faces bounds how well any model
that predicts psi/phi could ever match the velocity target.

Example:
    python scripts/visualize_helmholtz_dynamics.py \
        --file data/sim_0.hdf5 --start 0 --num-steps 40 --show
"""
import argparse
import json
from pathlib import Path

import numpy as np
import h5py as h5
import torch
import matplotlib.pyplot as plt

from nucleus.physics.poisson import (
    helmholtz_from_faces,
    reconstruct_velocity_from_helmholtz,
)

# Field names on disk (staggered faces + the interface for context).
DFUN = "dfun"
VELFACEX = "velfacex"
VELFACEY = "velfacey"


def grid_spacing(sim_params: dict) -> tuple[float, float]:
    """dx, dy from the sim params, matching the dataset's convention."""
    dx = (sim_params["x_max"] - sim_params["x_min"]) / (
        sim_params["num_blocks_x"] * int(sim_params["nx_block"])
    )
    dy = (sim_params["y_max"] - sim_params["y_min"]) / (
        sim_params["num_blocks_y"] * int(sim_params["ny_block"])
    )
    return dx, dy


def load_trajectory(filename: Path, start: int, num_steps: int, stride: int):
    """Read a window of face velocities and the interface from the HDF5 file, plus
    the grid spacing from the matching JSON."""
    time_slice = slice(start, start + num_steps * stride, stride)
    with h5.File(filename, "r") as handle:
        velfacex = torch.tensor(handle[VELFACEX][time_slice], dtype=torch.float32)
        velfacey = torch.tensor(handle[VELFACEY][time_slice], dtype=torch.float32)
        sdf = torch.tensor(handle[DFUN][time_slice], dtype=torch.float32)

    with open(str(filename).replace(".hdf5", ".json"), "r", encoding="utf-8") as handle:
        sim_params = json.load(handle)
    dx, dy = grid_spacing(sim_params)
    return velfacex, velfacey, sdf, dx, dy


def normalized_step_increment_rms(field: torch.Tensor) -> np.ndarray:
    """Per-step increment RMS of a ``(T, ...)`` field, normalized by the field's
    own std over the whole window so fields of different magnitude are comparable.
    A small, flat curve means the field evolves smoothly and predictably in time."""
    scale = field.std().clamp_min(1e-12)
    increments = (field[1:] - field[:-1]) / scale
    spatial_dims = tuple(range(1, increments.ndim))
    return increments.pow(2).mean(dim=spatial_dims).sqrt().numpy()


def radial_power_spectrum(field_2d: np.ndarray, dx: float, dy: float):
    """Radially averaged power spectrum of a 2D field. Returns (wavenumber,
    power); wavenumber is in cycles per domain unit so grids of different size
    line up on the same axis."""
    height, width = field_2d.shape
    spectrum = np.fft.fft2(field_2d - field_2d.mean())
    power = (np.abs(spectrum) ** 2) / (height * width)

    wavenumber_x = np.fft.fftfreq(width, d=dx)
    wavenumber_y = np.fft.fftfreq(height, d=dy)
    grid_x, grid_y = np.meshgrid(wavenumber_x, wavenumber_y)
    wavenumber = np.sqrt(grid_x**2 + grid_y**2).ravel()

    num_bins = min(height, width) // 2
    bins = np.linspace(0.0, wavenumber.max(), num_bins + 1)
    bin_index = np.clip(np.digitize(wavenumber, bins) - 1, 0, num_bins - 1)
    power_sum = np.bincount(bin_index, weights=power.ravel(), minlength=num_bins)
    counts = np.bincount(bin_index, minlength=num_bins)
    radial_power = power_sum[:num_bins] / np.maximum(counts[:num_bins], 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, radial_power


def averaged_spectrum(field: torch.Tensor, dx: float, dy: float):
    """Radial power spectrum averaged over the time window (reduces per-frame
    noise). ``field`` is ``(T, H, W)``."""
    spectra = [radial_power_spectrum(frame.numpy(), dx, dy) for frame in field]
    wavenumber = spectra[0][0]
    power = np.mean([spec for _, spec in spectra], axis=0)
    return wavenumber, power


def high_frequency_fraction(wavenumber: np.ndarray, power: np.ndarray) -> float:
    """Fraction of spectral power above the median wavenumber -- a scalar summary
    of how small-scale a field is (velocity high, potentials low)."""
    midpoint = wavenumber[len(wavenumber) // 2]
    total = power.sum()
    if total <= 0:
        return float("nan")
    return float(power[wavenumber >= midpoint].sum() / total)


def print_reconstruction_floor(velfacex, velfacey, psi, phi, dx, dy):
    """Round-trip the faces through the decomposition and report the normalized
    MAE -- the best a psi/phi-predicting model could do on the velocity target
    (before the model's extra band-gating / wall windowing)."""
    recon_x, recon_y = reconstruct_velocity_from_helmholtz(psi, phi, dx, dy)
    scale_x = velfacex.std().clamp_min(1e-12)
    scale_y = velfacey.std().clamp_min(1e-12)
    mae_x = ((recon_x - velfacex).abs().mean() / scale_x).item()
    mae_y = ((recon_y - velfacey).abs().mean() / scale_y).item()
    print("solver round-trip floor (normalized MAE of reconstructed velocity):")
    print(f"  velx: {mae_x:.2e}   vely: {mae_y:.2e}")
    print("  (near-zero => decomposition is consistent; the velocity-loss spike is")
    print("   derivative amplification, not a reconstruction floor)")


def plot(sdf, speed, psi, phi, increments, spectra, snapshot_index, output_path, show):
    figure = plt.figure(figsize=(15, 10))
    grid = figure.add_gridspec(3, 3, height_ratios=[1.1, 1.0, 1.0])

    # Row 1: snapshots of the two potentials and the speed, with interface contour.
    snapshot_axes = [figure.add_subplot(grid[0, col]) for col in range(3)]
    panels = [
        ("streamfunction psi", psi[snapshot_index].numpy(), "RdBu_r"),
        ("potential phi", phi[snapshot_index].numpy(), "RdBu_r"),
        ("speed |vel|", speed[snapshot_index].numpy(), "viridis"),
    ]
    interface = sdf[snapshot_index].numpy()
    for axis, (title, field, cmap) in zip(snapshot_axes, panels):
        limit = np.abs(field).max() or 1.0
        kwargs = dict(origin="lower", aspect="equal")
        if cmap == "RdBu_r":
            kwargs.update(vmin=-limit, vmax=limit)
        handle = axis.imshow(field, cmap=cmap, **kwargs)
        axis.contour(interface, levels=[0.0], colors="k", linewidths=0.8)
        axis.set_title(f"{title}  (t={snapshot_index})")
        figure.colorbar(handle, ax=axis, fraction=0.046)

    # Row 2: normalized per-step increment RMS over time -- temporal continuity.
    increment_axis = figure.add_subplot(grid[1, :])
    for name, curve in increments.items():
        increment_axis.plot(range(1, len(curve) + 1), curve, marker=".", label=name)
    increment_axis.set_xlabel("timestep")
    increment_axis.set_ylabel("per-step increment RMS\n(fraction of field std)")
    increment_axis.set_title(
        "Temporal continuity: how much each field changes per step "
        "(lower = smoother in time)"
    )
    increment_axis.legend(ncol=4)
    increment_axis.grid(True, alpha=0.3)

    # Row 3: radially averaged power spectrum -- where each field's variance lives.
    spectrum_axis = figure.add_subplot(grid[2, :])
    for name, (wavenumber, power) in spectra.items():
        spectrum_axis.loglog(wavenumber[1:], power[1:], label=name)
    spectrum_axis.set_xlabel("wavenumber (cycles per domain unit)")
    spectrum_axis.set_ylabel("radial power")
    spectrum_axis.set_title(
        "Spatial spectrum: potentials concentrate power at low k, "
        "velocity extends to high k"
    )
    spectrum_axis.legend(ncol=4)
    spectrum_axis.grid(True, alpha=0.3, which="both")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    print(f"saved figure to {output_path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--file", type=Path, required=True, help="trajectory HDF5 file")
    parser.add_argument("--start", type=int, default=0, help="first timestep to load")
    parser.add_argument("--num-steps", type=int, default=40, help="number of frames")
    parser.add_argument("--stride", type=int, default=1, help="timestep stride")
    parser.add_argument("--snapshot", type=int, default=None,
                        help="frame index (within the window) for the snapshot row; "
                             "defaults to the middle frame")
    parser.add_argument("--output", type=Path, default=Path("helmholtz_dynamics.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    velfacex, velfacey, sdf, dx, dy = load_trajectory(
        args.file, args.start, args.num_steps, args.stride
    )
    print(f"loaded {velfacex.shape[0]} frames from {args.file}  (dx={dx:.4g}, dy={dy:.4g})")

    # psi (T, H+1, W+1) nodal, phi (T, H, W) cell-centered.
    psi, phi = helmholtz_from_faces(velfacex, velfacey, dx, dy)
    # Cell-centered speed for display: average the faces onto centers.
    velx_centers = 0.5 * (velfacex[..., :-1] + velfacex[..., 1:])
    vely_centers = 0.5 * (velfacey[..., :-1, :] + velfacey[..., 1:, :])
    speed = torch.sqrt(velx_centers**2 + vely_centers**2)

    print_reconstruction_floor(velfacex, velfacey, psi, phi, dx, dy)

    increments = {
        "velx": normalized_step_increment_rms(velfacex),
        "vely": normalized_step_increment_rms(velfacey),
        "psi": normalized_step_increment_rms(psi),
        "phi": normalized_step_increment_rms(phi),
    }
    print("mean per-step increment RMS (fraction of field std):")
    for name, curve in increments.items():
        print(f"  {name:<5} {curve.mean():.3f}")

    spectra = {
        "velx": averaged_spectrum(velx_centers, dx, dy),
        "vely": averaged_spectrum(vely_centers, dx, dy),
        "psi": averaged_spectrum(psi, dx, dy),
        "phi": averaged_spectrum(phi, dx, dy),
    }
    print("high-wavenumber power fraction (above median k):")
    for name, (wavenumber, power) in spectra.items():
        print(f"  {name:<5} {high_frequency_fraction(wavenumber, power):.3f}")

    snapshot_index = args.snapshot if args.snapshot is not None else velfacex.shape[0] // 2
    plot(sdf, speed, psi, phi, increments, spectra, snapshot_index, args.output, args.show)


if __name__ == "__main__":
    main()
