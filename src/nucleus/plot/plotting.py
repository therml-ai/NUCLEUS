from calendar import c
from dataclasses import dataclass
from re import M
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, SymLogNorm, LogNorm, LinearSegmentedColormap
import seaborn as sns
from typing import Optional, Tuple
from nucleus.utils.physical_metrics import vorticity
from nucleus.test import TestResults
import joblib
import numpy as np

def ax_default(ax, title: Optional[str] = None):
    if title is not None:
        ax.set_title(title)
    #ax.axis("off")

def sdf_cmap():
    ranges = [0.0, 0.49, 0.51, 1]
    color_codes = ["blue", "white", "red"]
    colors = list(zip(ranges, color_codes))
    cmap = LinearSegmentedColormap.from_list('temperature_colormap', colors)
    return cmap

def plot_sdf(ax, sdf: torch.Tensor, title: Optional[str] = None):
    assert sdf.dim() == 2, "SDF must be a 2D tensor (H, W)"
    sdf = sdf.detach().cpu().numpy()
    norm = TwoSlopeNorm(vcenter=0, vmin=-6)
    im = ax.imshow(sdf, cmap="RdYlBu", norm=norm)
    ax.contour(sdf, colors="white", linewidths=0.5)
    ax.contour(sdf, levels=[0], colors="black", linewidths=0.5)
    ax_default(ax, title)
    return im

def temp_cmap():
    temp_ranges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.134, 0.167,
                    0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    color_codes = ['#0000FF', '#0443FF', '#0E7AFF', '#16B4FF', '#1FF1FF', '#21FFD3',
                   '#22FF9B', '#22FF67', '#22FF15', '#29FF06', '#45FF07', '#6DFF08',
                   '#9EFF09', '#D4FF0A', '#FEF30A', '#FEB709', '#FD7D08', '#FC4908',
                   '#FC1407', '#FB0007']
    colors = list(zip(temp_ranges, color_codes))
    cmap = LinearSegmentedColormap.from_list('temperature_colormap', colors)
    return cmap

def plot_temp(ax, temp: torch.Tensor, bulk_temp, heater_temp, title: Optional[str] = None):
    assert temp.dim() == 2, "Temp must be a 2D tensor (H, W)"
    temp = temp.detach().cpu().numpy()
    norm = Normalize(vmin=bulk_temp, vmax=heater_temp, clip=True)
    im = ax.imshow(temp, cmap=temp_cmap(), norm=norm)
    ax_default(ax, title)
    return im

def plot_vel_mag(ax, vel_mag: torch.Tensor, title: Optional[str] = None):
    assert vel_mag.dim() == 2, "Vel mag must be a 2D tensor (H, W)"
    vel_mag = vel_mag.detach().cpu().numpy()
    index = int(0.99 * vel_mag.size)
    vmax = np.sort(vel_mag.flatten())[index]
    im = ax.imshow(vel_mag, cmap="rocket", vmin=0, vmax=vmax)
    ax_default(ax, title)
    return im

def plot_vorticity(ax, vorticity: torch.Tensor, min_vort=None, max_vort=None, title: Optional[str] = None):
    assert vorticity.dim() == 2, "Vorticity must be a 2D tensor (H, W)"
    vorticity = vorticity.detach().cpu().numpy()
    # use a diverging colormap, centered at 0.
    norm = SymLogNorm(linthresh=0.5, vmin=min_vort, vmax=max_vort)
    im = ax.imshow(vorticity, cmap="icefire", norm=norm)
    ax_default(ax, title)
    return im

def plot_rollout_stability(save_dir: str, pred_rollout: torch.Tensor, target_rollout: torch.Tensor):
    sdf = pred_rollout[:, 0, :, :]
    temp = pred_rollout[:, 1, :, :]
    velx = pred_rollout[:, 2, :, :]
    vely = pred_rollout[:, 3, :, :]
    target_sdf = target_rollout[:, 0, :, :]
    target_temp = target_rollout[:, 1, :, :]
    target_velx = target_rollout[:, 2, :, :]
    target_vely = target_rollout[:, 3, :, :]
    
    # Rollout stability of predicted and target rollouts (i.e. the norm of the differences between consecutive timesteps)
    fig, axs = plt.subplots(1, 4, figsize=(10, 5), layout="constrained")
    
    sdf_steps = torch.norm(sdf[1:] - sdf[:-1], dim=(-2, -1))
    temp_steps = torch.norm(temp[1:] - temp[:-1], dim=(-2, -1))
    velx_steps = torch.norm(velx[1:] - velx[:-1], dim=(-2, -1))
    vely_steps = torch.norm(vely[1:] - vely[:-1], dim=(-2, -1))
    
    target_sdf_steps = torch.norm(target_sdf[1:] - target_sdf[:-1], dim=(-2, -1))
    target_temp_steps = torch.norm(target_temp[1:] - target_temp[:-1], dim=(-2, -1))
    target_velx_steps = torch.norm(target_velx[1:] - target_velx[:-1], dim=(-2, -1))
    target_vely_steps = torch.norm(target_vely[1:] - target_vely[:-1], dim=(-2, -1))
    
    print(sdf_steps.shape, target_sdf_steps.shape)

    axs[0].plot(sdf_steps, label="Predicted")
    axs[0].plot(target_sdf_steps, label="Target")
    axs[1].plot(temp_steps, label="Predicted")
    axs[1].plot(target_temp_steps, label="Target")
    axs[2].plot(velx_steps, label="Predicted")
    axs[2].plot(target_velx_steps, label="Target")
    axs[3].plot(vely_steps, label="Predicted")
    axs[3].plot(target_vely_steps, label="Target")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[0].set_title("SDF")
    axs[1].set_title("Temperature")
    axs[2].set_title("Velocity X")
    axs[3].set_title("Velocity Y")
    axs[0].set_xlabel("Time")
    axs[1].set_xlabel("Time")
    axs[2].set_xlabel("Time")
    axs[3].set_xlabel("Time")
    axs[0].set_ylabel("||r_t - r_{t-1}||_2")
    plt.savefig(f"{save_dir}/stability.png")
    plt.close()

def plot_rollout(
    save_dir: str,
    rollout: torch.Tensor,
    test_results: TestResults,
    step_size: int
):
    def make_plot(timestep, rollout):
        sdf = torch.flipud(rollout[timestep, 0, :, :])
        temp = torch.flipud(rollout[timestep, 1, :, :])
        velx = torch.flipud(rollout[timestep, 2, :, :])
        vely = torch.flipud(rollout[timestep, 3, :, :])
        vel_mag = torch.sqrt(velx**2 + vely**2)
        vort = vorticity(velx, vely, 1/4, 1/4)
        
        fig, axs = plt.subplots(1, 4, figsize=(10, 5), layout="constrained")
        
        # 1. Plot the SDF
        im = plot_sdf(axs[0], sdf)
        plt.colorbar(im, ax=axs[0], fraction=0.04, pad=0.05)
        
        # 2. Plot the temperature
        im = plot_temp(axs[1], temp, test_results.fluid_params["bulk_temp"], test_results.fluid_params["heater"]["wallTemp"])
        plt.colorbar(im, ax=axs[1], fraction=0.04, pad=0.05)
        
        # 3. Plot the velocity magnitude
        im = plot_vel_mag(axs[2], vel_mag)
        plt.colorbar(im, ax=axs[2], fraction=0.04, pad=0.05)
        
        # 4. Plot the vorticity
        im = plot_vorticity(axs[3], vort)
        plt.colorbar(im, ax=axs[3], fraction=0.04, pad=0.05)
        
        plt.savefig(f"{save_dir}/rollout_{str(timestep).zfill(4)}.png", bbox_inches="tight")
        plt.close()

    # TODO: really don't want batch dimension, but some of the metrics are
    # pretty rigid about using a batch dimension...
    rollout = rollout.squeeze(0)
    assert rollout.dim() == 4, "Rollout must be a 4D tensor (T, C, H, W)"
    
    joblib.Parallel(n_jobs=4)(joblib.delayed(make_plot)(timestep, rollout) for timestep in range(0, rollout.shape[0], step_size))
    
def plot_rollout_moe_overlay(
    save_dir: str,
    rollout: torch.Tensor,
    test_results: TestResults,
    step_size: int
):
    def make_plot(timestep, rollout):
        temp = torch.flipud(rollout[timestep, 1, :, :])
        velx = torch.flipud(rollout[timestep, 2, :, :])
        vely = torch.flipud(rollout[timestep, 3, :, :])
        vel_mag = torch.sqrt(velx**2 + vely**2)
        
        # Model processes five frames at a time, so divide by 5 to get correct index into model output.
        moe_output = test_results.moe_outputs[timestep // 5]
        
        fig, axs = plt.subplots(moe_output.num_experts, 2, figsize=(5, 10), layout="constrained")

        assert moe_output.topk_indices.dim() == 5, "Topk indices must be of shape (1, T, H, W, topk)"
        assert moe_output.topk_indices.shape[0] == 1, "Topk indices must be of shape (1, T, H, W, topk)"
        topk_indices = moe_output.topk_indices.squeeze(0)[timestep % 5] # get the correct timestep from the model output.

        print(moe_output.tokens_per_expert)
        for expert_id, axs_row in enumerate(axs):
            ax_temp = axs_row[0]
            ax_vel = axs_row[1]
            im_temp = ax_temp.imshow(temp, cmap=temp_cmap())
            
            index = int(0.99 * vel_mag.numel())
            vmax = np.sort(vel_mag.view(-1))[index]
            im_vel_mag = ax_vel.imshow(vel_mag, cmap="rocket", vmin=0, vmax=vmax)
            
            expert_map = (topk_indices == expert_id).sum(dim=-1).float()
            expert_overlay = torch.nn.functional.interpolate(expert_map.view(1, 1, 16, 16), size=(64, 64), mode="nearest").squeeze()
            expert_overlay = torch.flipud(expert_overlay.bool())
            
            ax_temp.imshow(expert_overlay, cmap="gray", vmin=0, vmax=1, alpha=0.7 * (expert_overlay > 0))
            ax_vel.imshow(expert_overlay, cmap="gray", vmin=0, vmax=1, alpha=0.7 * (expert_overlay > 0))
            
            ax_temp.axis("off")
            ax_vel.axis("off")
            
            plt.savefig(f"{save_dir}/rollout_{str(timestep).zfill(4)}.png", bbox_inches="tight")
        plt.close()
    
    joblib.Parallel(n_jobs=-1)(joblib.delayed(make_plot)(timestep, rollout.squeeze(0)) for timestep in range(0, rollout.shape[1], step_size))
    
if __name__ == "__main__":
    import h5py
    import numpy as np
    from nucleus.utils.physical_metrics import vorticity
    with h5py.File("/share/crsp/lab/amowli/share/BubbleML_2_downsampled64_bicubic/PoolBoiling-Subcooled-OP250-2D/Twall_97.hdf5", "r") as f:
        sdf = torch.from_numpy(f["dfun"][400])
        temp = torch.from_numpy(f["temperature"][400])
        velx = torch.from_numpy(f["velx"][400])
        vely = torch.from_numpy(f["vely"][400])

    print(temp.min(), temp.max())

    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
    plot_sdf(ax, torch.flipud(sdf))
    plt.savefig("test_subcooled_sdf.pdf", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
    plot_temp(ax, torch.flipud(temp), 50, 97)
    plt.savefig("test_subcooled_temp.pdf", bbox_inches="tight")
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
    plot_vel_mag(ax, torch.flipud(torch.sqrt(velx**2 + vely**2)))
    plt.savefig("test_subcooled_velx.pdf", bbox_inches="tight")
    plt.close()
    
    del sdf, temp, velx, vely
    
    with h5py.File("/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Saturated-FC72-2D/Twall_104.hdf5", "r") as f:
        sat_sdf = torch.from_numpy(f["dfun"][550])
        sat_temp = torch.from_numpy(f["temperature"][550])
        sat_velx = torch.from_numpy(f["velx"][550])
        sat_vely = torch.from_numpy(f["vely"][550])
        
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
    plot_sdf(ax, torch.flipud(sat_sdf))
    plt.savefig("test_saturated_sdf.pdf", bbox_inches="tight")
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
    plot_temp(ax, torch.flipud(sat_temp), 58, 104)
    plt.savefig("test_saturated_temp.pdf", bbox_inches="tight")
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
    plot_vel_mag(ax, torch.flipud(torch.sqrt(sat_velx**2 + sat_vely**2)))
    plt.savefig("test_saturated_velx.pdf", bbox_inches="tight")
    plt.close()
    
    del sat_sdf, sat_temp, sat_velx, sat_vely
    """

    vel_mag = torch.sqrt(velx**2 + vely**2)
    v = vorticity(velx, vely, 1/32, 1/32)
    
    timestep = 400
    fig, axs = plt.subplots(1, 4, figsize=(10, 3), layout="constrained")
    t = plot_sdf(axs[0], torch.flipud(sdf))
    axs[0].set_title("Distance to Interface")
    plt.colorbar(t, ax=axs[0], ticks=[-8, -4, 0, 0.1, 0.4], fraction=0.05, pad=0.05)
    t = plot_temp(axs[1], torch.flipud(temp), 41, 93)
    axs[1].set_title("Temperature")
    plt.colorbar(t, ax=axs[1], fraction=0.05, pad=0.05)
    t = plot_vel_mag(axs[2], torch.flipud(vel_mag))
    axs[2].set_title("Velocity Magnitude")
    plt.colorbar(t, ax=axs[2], fraction=0.05, pad=0.05)
    t = plot_vorticity(axs[3], torch.flipud(v))
    axs[3].set_title("Vorticity")
    plt.colorbar(t, ax=axs[3], ticks=[-10, 0, 10], fraction=0.05, pad=0.05)
    plt.savefig("test.pdf", bbox_inches="tight")
    