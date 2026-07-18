import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

def load_trajectory(path):
    FIELDS = ["dfun", "temperature", "velx", "vely"]
    with h5py.File(path, "r") as handle:
        return np.concatenate([handle[field] for field in FIELDS], axis=-1)

def metrics(array: np.array):
    pass

def vapor_volume(sdf, dx, dy):
    vapor_mask = sdf > 0
    return vapor_mask.astype(float).sum(axis=(-2, -1)) * dx * dy

def vapor_volume_at_height(sdf, dx, dy):
    vapor_mask = sdf > 0
    return vapor_mask.astype(float).sum(axis=(-1)) * dx

def plot_vapor_volume_at_height(pred_vvh, gt_vvh, save_path):
    # pred_vvh and gt_vvh have shape (time, height); transpose so imshow puts
    # height on the y-axis and time on the x-axis, with a shared color scale so
    # prediction and ground truth are directly comparable.
    vmin = min(pred_vvh.min(), gt_vvh.min())
    vmax = max(pred_vvh.max(), gt_vvh.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, data, title in zip(axes, (gt_vvh, pred_vvh), ("Ground Truth", "Prediction")):
        image = ax.imshow(
            data.T,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("time step")
    axes[0].set_ylabel("height (cell index)")
    fig.colorbar(image, ax=axes, label="vapor volume at height")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_time_averaged_vapor_volume(pred_vvh, gt_vvh, save_path):
    # Average the (time, height) vapor volume over time into a profile vs height,
    # with height on the y-axis to match the imshow orientation.
    heights = np.arange(pred_vvh.shape[1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(gt_vvh.mean(axis=0), heights, label="Ground Truth")
    ax.plot(pred_vvh.mean(axis=0), heights, label="Prediction")
    ax.set_xlabel("time-averaged vapor volume at height")
    ax.set_ylabel("height (cell index)")
    ax.legend()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _flatten_for_kde(array, max_samples, rng):
    # Kernel-density estimation cost grows with the sample count, so subsample
    # large trajectories (reproducibly) before fitting.
    values = array.ravel()
    if values.size > max_samples:
        values = values[rng.choice(values.size, size=max_samples, replace=False)]
    return values

def plot_temperature_density(pred_temp, gt_temp, save_path, num_points=512, max_samples=500_000):
    # Overlay smooth kernel-density estimates of the temperature distributions,
    # evaluated on a shared grid so predicted and ground-truth are comparable.
    rng = np.random.default_rng(0)
    gt_values = _flatten_for_kde(gt_temp, max_samples, rng)
    pred_values = _flatten_for_kde(pred_temp, max_samples, rng)

    lo = min(pred_values.min(), gt_values.min())
    hi = max(pred_values.max(), gt_values.max())
    grid = np.linspace(lo, hi, num_points)

    fig, ax = plt.subplots(figsize=(7, 5))
    for values, label in ((gt_values, "Ground Truth"), (pred_values, "Prediction")):
        density = gaussian_kde(values)(grid)
        line, = ax.plot(grid, density, label=label)
        ax.fill_between(grid, density, alpha=0.3, color=line.get_color())
    ax.set_yscale("log")
    ax.set_xlabel("temperature")
    ax.set_ylabel("density")
    ax.legend()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    args = parser.parse_args()
    path = Path(args.path)
    pred = load_trajectory(path / "pred_trajectory.hdf5")
    gt = load_trajectory(path / "gt_trajectory.hdf5")

    pred_vv = vapor_volume(pred[..., 0], 1/32, 1/32).mean()
    gt_vv = vapor_volume(gt[..., 0], 1/32, 1/32).mean()

    print(pred_vv, gt_vv)

    pred_vvh = vapor_volume_at_height(pred[..., 0], 1/32, 1/32)
    gt_vvh = vapor_volume_at_height(gt[..., 0], 1/32, 1/32)

    save_path = path / "vapor_volume_at_height.png"
    plot_vapor_volume_at_height(pred_vvh, gt_vvh, save_path)
    print(f"saved vapor-volume-at-height plot to {save_path}")

    time_avg_save_path = path / "time_averaged_vapor_volume.png"
    plot_time_averaged_vapor_volume(pred_vvh, gt_vvh, time_avg_save_path)
    print(f"saved time-averaged vapor-volume plot to {time_avg_save_path}")

    temp_density_save_path = path / "temperature_density.png"
    plot_temperature_density(pred[..., 1], gt[..., 1], temp_density_save_path)
    print(f"saved temperature-density plot to {temp_density_save_path}")

if __name__ == "__main__":
    main()