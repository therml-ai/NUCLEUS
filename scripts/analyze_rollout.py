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

def velocity_divergence(velx, vely, dx, dy):
    # div(u) = d velx/dx + d vely/dy. In the (T, H, W) layout width (x) is the last
    # axis and height (y) the second-to-last.
    dvelx_dx = np.gradient(velx, dx, axis=-1)
    dvely_dy = np.gradient(vely, dy, axis=-2)
    return dvelx_dx + dvely_dy

def max_divergence_over_time(velx, vely, sdf, dx, dy):
    # Worst-case divergence magnitude of liquid bulk  at each timestep.
    divergence = velocity_divergence(velx, vely, dx, dy)

    return (np.abs(divergence) * (sdf < -1).astype(float)).max(axis=(-2, -1))

def mac_divergence(velx, vely, dx, dy):
    # Staggered (MAC) divergence: adjacent one-sided differences of face-valued
    # velocities, co-located at interior cell centers. This uses the "each cell owns
    # its east/north face" convention; the west/south convention just crops the
    # other side (see _staggered_divergence_variants).
    dvelx_dx = np.diff(velx, axis=-1) / dx   # (..., H, W-1)
    dvely_dy = np.diff(vely, axis=-2) / dy   # (..., H-1, W)
    return dvelx_dx[..., 1:, :] + dvely_dy[..., :, 1:]   # cells (1..H-1, 1..W-1)

def _staggered_divergence_variants(sdf, velx, vely, dx, dy):
    # Both MAC face-ownership conventions. Each returns (divergence, matching sdf)
    # co-located on the cells the convention resolves.
    dvelx_dx = np.diff(velx, axis=-1) / dx   # (..., H, W-1)
    dvely_dy = np.diff(vely, axis=-2) / dy   # (..., H-1, W)
    east_north = (dvelx_dx[..., 1:, :] + dvely_dy[..., :, 1:], sdf[..., 1:, 1:])
    west_south = (dvelx_dx[..., :-1, :] + dvely_dy[..., :, :-1], sdf[..., :-1, :-1])
    return {"east/north": east_north, "west/south": west_south}

def _staggered_divergence_4th(sdf, velx, vely, dx, dy):
    # Fourth-order-accurate staggered divergence. The first derivative of the
    # face-valued velocity at a cell center is
    #   d f/dx ~ [27 (f[j] - f[j-1]) - (f[j+1] - f[j-2])] / (24 dx),
    # which cancels the O(dx^2) truncation of the adjacent difference. Both
    # face-ownership conventions (like the second-order variants) co-located at
    # interior cell centers, losing two cells at each boundary. If the ~1e-2 from
    # the adjacent-difference variants is really discretization truncation on
    # smooth, nearly divergence-free data, this should be much smaller.
    dvelx_dx = (27 * (velx[..., 2:-1] - velx[..., 1:-2]) - (velx[..., 3:] - velx[..., :-3])) / (24 * dx)   # (..., H, W-3)
    dvely_dy = (27 * (vely[..., 2:-1, :] - vely[..., 1:-2, :]) - (vely[..., 3:, :] - vely[..., :-3, :])) / (24 * dy)  # (..., H-3, W)
    east_north = (dvelx_dx[..., 2:-1, :] + dvely_dy[..., :, 2:-1], sdf[..., 2:-1, 2:-1])
    west_south = (dvelx_dx[..., 1:-2, :] + dvely_dy[..., :, 1:-2], sdf[..., 1:-2, 1:-2])
    return {"east/north (4th)": east_north, "west/south (4th)": west_south}

def verify_mac_divergence(gt, dx, dy, liquid_sdf=-2.0, percentile=99):
    # Is the ground-truth velocity face-valued (staggered) or cell-centered?
    # Compare the staggered (adjacent-diff) divergence against the collocated
    # (central-diff) one in the liquid bulk, where incompressible flow should be
    # divergence-free so real interface (Stefan) divergence does not pollute the
    # test. A high percentile (not the max) is used so a few outlier spikes at
    # bubble events do not dominate the comparison. Both MAC conventions are
    # tried; the smaller one wins.
    sdf, velx, vely = gt[..., 0], gt[..., 2], gt[..., 3]

    def liquid_percentile(divergence, div_sdf):
        values = np.abs(divergence)[div_sdf < liquid_sdf]
        return float(np.percentile(values, percentile)) if values.size else float("nan")

    central = liquid_percentile(velocity_divergence(velx, vely, dx, dy), sdf)
    staggered = _staggered_divergence_variants(sdf, velx, vely, dx, dy)
    staggered.update(_staggered_divergence_4th(sdf, velx, vely, dx, dy))
    variants = {
        name: liquid_percentile(divergence, div_sdf)
        for name, (divergence, div_sdf) in staggered.items()
    }
    best_name = min(variants, key=variants.get)
    mac = variants[best_name]
    ratio = mac / central if central else float("nan")

    print(f"ground-truth liquid-bulk divergence, p{percentile} |div| (should be ~0 for incompressible flow):")
    for name, value in variants.items():
        print(f"  staggered  ({name}) = {value:.4e}")
    print(f"  collocated (central   ) = {central:.4e}")
    print(f"  best staggered / collocated = {ratio:.4e}  ({best_name})")
    if ratio < 0.05:
        print(f"  -> velocity looks FACE-VALUED (MAC, {best_name}): use adjacent differences for the curl.")
    else:
        print("  -> velocity looks CELL-CENTERED: central differences are the right match.")

def plot_vapor_volume_at_height(pred_vvh, gt_vvh, save_path):
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

def plot_max_divergence(pred_max_div, gt_max_div, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(gt_max_div.shape[0]), gt_max_div, label="Ground Truth")
    ax.plot(range(pred_max_div.shape[0]), pred_max_div, label="Prediction")
    ax.set_xlabel("time step")
    ax.set_ylabel("max |divergence|")
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

    verify_mac_divergence(gt, 1/32, 1/32)

    pred_vv = vapor_volume(pred[..., 0], 1/32, 1/32).mean()
    gt_vv = vapor_volume(gt[..., 0], 1/32, 1/32).mean()

    print("Vapor Volume:")
    print("Predicted: ", pred_vv)
    print("Ground-truth: ", gt_vv)

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

    pred_max_div = max_divergence_over_time(pred[..., 2], pred[..., 3], pred[..., 0], 1/32, 1/32)
    gt_max_div = max_divergence_over_time(gt[..., 2], gt[..., 3], gt[..., 0], 1/32, 1/32)

    max_div_save_path = path / "max_divergence.png"
    plot_max_divergence(pred_max_div, gt_max_div, max_div_save_path)
    print(f"saved max-divergence plot to {max_div_save_path}")

if __name__ == "__main__":
    main()