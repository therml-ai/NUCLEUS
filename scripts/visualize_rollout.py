import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from pathlib import Path
import torch

from boiling_viz.boiling_video import BoilingVideoBuilder

from nucleus.physics.poisson import divergence_centers_from_faces


def _read_field(handle, key):
    # inf.py's collocated save keeps a trailing singleton channel; the natural-grid
    # (divfree) save does not. Drop a trailing size-1 axis so every field is
    # (T, H, W...).
    array = handle[key][:]
    if array.shape[-1] == 1:
        array = array[..., 0]
    return array


def load_trajectory(path):
    # Divfree rollouts save the velocity on its staggered faces (velfacex/velfacey);
    # collocated rollouts save it cell-centered (velx/vely). Detect which and flag it
    # so downstream operators use the matching stencil.
    with h5py.File(path, "r") as handle:
        is_face = "velfacex" in handle
        velx_key, vely_key = ("velfacex", "velfacey") if is_face else ("velx", "vely")
        return {
            "sdf": _read_field(handle, "dfun"),
            "temperature": _read_field(handle, "temperature"),
            "velx": _read_field(handle, velx_key),
            "vely": _read_field(handle, vely_key),
            "is_face": is_face,
        }


def faces_to_cell_velocity(velx, vely):
    # Average staggered face velocities onto cell centers (..., H, W).
    velx_cell = 0.5 * (velx[..., :-1] + velx[..., 1:])
    vely_cell = 0.5 * (vely[..., :-1, :] + vely[..., 1:, :])
    return velx_cell, vely_cell


def cell_velocity(data):
    # Velocities on cell centers, averaging the faces first for divfree rollouts.
    if data["is_face"]:
        return faces_to_cell_velocity(data["velx"], data["vely"])
    return data["velx"], data["vely"]


def velocity_divergence(velx, vely, is_face, dx, dy):
    # div(u) = d velx/dx + d vely/dy. On staggered faces this is the exact
    # cell-centered operator; on collocated velocities it is a central difference. In
    # the (..., H, W) layout width (x) is the last axis and height (y) the second-last.
    if is_face:
        divergence = divergence_centers_from_faces(
            torch.as_tensor(velx), torch.as_tensor(vely), dx, dy
        )
        return divergence.numpy()
    return np.gradient(velx, dx, axis=-1) + np.gradient(vely, dy, axis=-2)


def plot_divergence(data, save_path, num_frames=6, dx=1/32, dy=1/32):
    # Velocity divergence at evenly-spaced frames, laid out across columns. It is signed
    # and spans orders of magnitude, so a symmetric-log norm with a single scale shared
    # across the frames is used. The face stencil is used when the velocity is staggered.
    divergence = velocity_divergence(data["velx"], data["vely"], data["is_face"], dx, dy)

    frame_indices = np.linspace(0, divergence.shape[0] - 1, num_frames).astype(int)
    div_vmax = max(float(np.abs(divergence[frame_indices]).max()), 1e-12)
    norm = SymLogNorm(linthresh=div_vmax * 1e-6, vmin=-div_vmax, vmax=div_vmax, base=10)

    fig, axes = plt.subplots(1, num_frames, figsize=(3 * num_frames, 3), squeeze=False)
    for col, frame_idx in enumerate(frame_indices):
        ax = axes[0][col]
        image = ax.imshow(divergence[frame_idx], origin="lower", cmap="RdBu_r", norm=norm)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {frame_idx}")
    fig.colorbar(image, ax=list(axes.flat), fraction=0.046)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def vapor_volume(sdf, dx, dy):
    vapor_mask = sdf > 0
    return vapor_mask.astype(float).sum(axis=(-2, -1)) * dx * dy

def heat_flux(temp, sdf, heater_temp, dx, dy):
    def denormalize_temp_grad(temp, t_wall, t_bulk=50, k=0.054):
        del_t = t_wall - t_bulk
        return 2 * k * del_t * (1 - temp)
    
    def non_dim_temp(temp, bulk_temp=50):
        return (temp - bulk_temp) / (heater_temp - bulk_temp)

    lc = 0.0007
    x_grid = torch.arange(-8, 8, dx) + dx / 2

    print(temp.min(), temp.max())

    liquid_mask = sdf < 0 #temp < 58

    temp = non_dim_temp(temp)
    d_temp = denormalize_temp_grad(temp[:, 0], heater_temp)
    heater_mask = (x_grid >= -5.25) & (x_grid <= 5.25)
    hflux_list = torch.mean((heater_mask[None, :] & liquid_mask[:, 0]).to(float) * d_temp / (dy * lc),
                            dim=1)
    hflux = torch.mean(hflux_list)
    qmax = torch.max(hflux_list)
    return hflux, qmax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    args = parser.parse_args()
    path = Path(args.path)
    pred = load_trajectory(path / "pred_trajectory.hdf5")
    gt = load_trajectory(path / "gt_trajectory.hdf5")

    print(pred["sdf"].shape, "faces" if pred["is_face"] else "cells")
    print(gt["sdf"].shape, "faces" if gt["is_face"] else "cells")

    # The video builder expects collocated (T, H, W, 4) fields, so average the faces
    # to cell centers for divfree rollouts.
    velx_cell, vely_cell = cell_velocity(pred)
    cell_trajectory = np.stack([pred["sdf"], pred["temperature"], velx_cell, vely_cell], axis=-1)

    builder = BoilingVideoBuilder(cell_trajectory[::10])
    builder.make_video(
        f"{path}/traj.gif",
        duration=10,
        colorbars=False,
        step_counter=True,
        field_titles=True,
        transparent_nan=False,
        columnwise=True
    )

    divergence_save_path = path / "divergence.png"
    plot_divergence(pred, divergence_save_path)
    print(f"saved divergence plot to {divergence_save_path}")

if __name__ == "__main__":
    main()