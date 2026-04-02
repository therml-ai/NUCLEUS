import torch
import matplotlib.pyplot as plt
from nucleus.plot.plotting import plot_sdf, plot_temp, plot_vel_mag
from scipy.stats import wasserstein_distance

op250_results = torch.load(
    "/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling64_op250_48123677/checkpoints/inference_rollouts/test_results_clip_boiling_point.pt",
    weights_only=False
)

for result in op250_results:
    preds = result.preds[:, :251:25].squeeze(0) 
    
    fig, axs = plt.subplots(3, preds.shape[0], figsize=(18, 5), layout="constrained")
    for idx, pred in enumerate(preds):
        sdf = torch.flipud(pred[0, :, :])
        temp = torch.flipud(pred[1, :, :])
        vel_mag = torch.flipud(torch.sqrt(pred[2, :, :]**2 + pred[3, :, :]**2))
        plot_sdf(axs[0, idx], sdf)
        plot_temp(axs[1, idx], temp, result.fluid_params["bulk_temp"], result.fluid_params["heater"]["wallTemp"])
        plot_vel_mag(axs[2, idx], vel_mag)
        axs[0, idx].set_title(f"Step {idx * 25}")
        axs[0, idx].set_xticks([])
        axs[0, idx].set_yticks([])
        axs[1, idx].set_xticks([])
        axs[1, idx].set_yticks([])
        axs[2, idx].set_xticks([])
        axs[2, idx].set_yticks([])
    plt.savefig(f"rollout_{result.case_name}.pdf", bbox_inches="tight")
    plt.close()
    
    sdf = result.preds.squeeze(0)[:, 0, :, :]
    temp = result.preds.squeeze(0)[:, 1, :, :]
    velx = result.preds.squeeze(0)[:, 2, :, :]
    vely = result.preds.squeeze(0)[:, 3, :, :]
    
    sdf_target = result.targets.squeeze(0)[:, 0, :, :]
    temp_target = result.targets.squeeze(0)[:, 1, :, :]
    velx_target = result.targets.squeeze(0)[:, 2, :, :]
    vely_target = result.targets.squeeze(0)[:, 3, :, :]
    
    temp_hist = torch.histogram(
        temp.flatten(), 
        bins=100, 
        range=(result.fluid_params["bulk_temp"], result.fluid_params["heater"]["wallTemp"]),
        density=True
    )
    velx_hist = torch.histogram(velx.flatten(), bins=100, range=(-3, 3), density=True)
    vely_hist = torch.histogram(vely.flatten(), bins=100, range=(-4, 4), density=True)

    temp_target_hist = torch.histogram(
        temp_target.flatten(), 
        bins=100, 
        range=(result.fluid_params["bulk_temp"], result.fluid_params["heater"]["wallTemp"]),
        density=True
    )
    velx_target_hist = torch.histogram(velx_target.flatten(), bins=100, range=(-3, 3), density=True)
    vely_target_hist = torch.histogram(vely_target.flatten(), bins=100, range=(-4, 4), density=True)
    
    temp_wd = wasserstein_distance(temp.flatten(), temp_target.flatten())
    velx_wd = wasserstein_distance(velx.flatten(), velx_target.flatten())
    vely_wd = wasserstein_distance(vely.flatten(), vely_target.flatten())
    
    fig, axs = plt.subplots(3, 1, figsize=(4, 5), layout="constrained")
    
    axs[0].plot(temp_hist.bin_edges[:-1], temp_hist.hist, label="Predicted")
    axs[0].fill_between(temp_hist.bin_edges[:-1], temp_hist.hist, alpha=0.5)
    axs[1].plot(velx_hist.bin_edges[:-1], velx_hist.hist, label="Predicted")
    axs[1].fill_between(velx_hist.bin_edges[:-1], velx_hist.hist, alpha=0.5)
    axs[2].plot(vely_hist.bin_edges[:-1], vely_hist.hist, label="Predicted")
    axs[2].fill_between(vely_hist.bin_edges[:-1], vely_hist.hist, alpha=0.5)

    axs[0].plot(temp_target_hist.bin_edges[:-1], temp_target_hist.hist, label="Target")
    axs[0].fill_between(temp_target_hist.bin_edges[:-1], temp_target_hist.hist, alpha=0.5)
    axs[1].plot(velx_target_hist.bin_edges[:-1], velx_target_hist.hist, label="Target")
    axs[1].fill_between(velx_target_hist.bin_edges[:-1], velx_target_hist.hist, alpha=0.5)
    axs[2].plot(vely_target_hist.bin_edges[:-1], vely_target_hist.hist, label="Target")
    axs[2].fill_between(vely_target_hist.bin_edges[:-1], vely_target_hist.hist, alpha=0.5)
    axs[0].set_yscale("log")
    
    axs[0].set_xlabel("Temperature")
    axs[1].set_xlabel("Velocity X")
    axs[2].set_xlabel("Velocity Y")
    
    axs[0].set_ylabel("Density")
    axs[1].set_ylabel("Density")
    axs[2].set_ylabel("Density")
    
    axs[0].set_title(f"OP250 Temp EMD: {temp_wd:.4f}")
    axs[1].set_title(f"OP250 Vel-x EMD: {velx_wd:.4f}")
    axs[2].set_title(f"OP250 Vel-y EMD: {vely_wd:.4f}")
    
    plt.legend()
    plt.savefig(f"op250_dist_{result.case_name}.pdf", bbox_inches="tight")
    plt.close()