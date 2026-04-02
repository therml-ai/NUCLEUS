import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import wasserstein_distance

def pretty_name(case_name: str):
    parts = case_name.split("_")
    parts[0] = parts[0].capitalize()[:3]
    parts[1] = parts[1].upper()
    parts[2] = str(int(float(parts[2])))
    parts.append("°C")
    return " ".join(parts)

def filter_sat(test_results):
    return [t for t in test_results if "sub" in t.case_name]

def plot_temp_dist(ax, preds, targets, bulk_temp, heater_temp):
    pred_temp = preds[:, :, 1, :, :]
    pred_sdf = preds[:, :, 0, :, :]
    target_temp = targets[:, :, 1, :, :]
    target_sdf = targets[:, :, 0, :, :]
    
    pred_liquid_mask = pred_sdf < 0
    target_liquid_mask = target_sdf < 0
    
    pred_temp_liquid = pred_temp[pred_liquid_mask]
    target_temp_liquid = target_temp[target_liquid_mask]
    
    emd = wasserstein_distance(pred_temp_liquid.flatten(), target_temp_liquid.flatten())
    print(f"EMD: {emd:.4f}")

    max_temp = target_temp.max()
    print(max_temp)
    
    pred_hist = torch.histogram(pred_temp_liquid.flatten(), bins=100, range=(bulk_temp, bulk_temp + 10), density=True)
    target_hist = torch.histogram(target_temp_liquid.flatten(), bins=100, range=(bulk_temp, bulk_temp + 10), density=True)
        
    ax.plot(pred_hist.bin_edges[:-1], pred_hist.hist, label="Predicted")
    ax.fill_between(pred_hist.bin_edges[:-1], pred_hist.hist, alpha=0.5)
    ax.plot(target_hist.bin_edges[:-1], target_hist.hist, label="Target")
    ax.fill_between(target_hist.bin_edges[:-1], target_hist.hist, alpha=0.5)
    ax.set_yscale("log")
    ax.set_title(f"EMD: {emd:.4f}")
    
def plot_vel_dist(ax, preds, targets):
    print(preds.shape)
    pred_velx = preds[:, :, 2, :, :]
    pred_vely = preds[:, :, 3, :, :]
    target_velx = targets[:, :, 2, :, :]
    target_vely = targets[:, :, 3, :, :]
    print(target_velx.min(), target_velx.max(), target_vely.min(), target_vely.max())
    
    pred_hist = torch.histogram(pred_vely.flatten(), bins=100, density=True)
    target_hist = torch.histogram(target_vely.flatten(), bins=100, density=True)
    
    emd = wasserstein_distance(pred_velx.flatten(), target_velx.flatten())
    
    ax.plot(pred_hist.bin_edges[:-1], pred_hist.hist, label="Predicted")
    ax.fill_between(pred_hist.bin_edges[:-1], pred_hist.hist, alpha=0.5)
    ax.plot(target_hist.bin_edges[:-1], target_hist.hist, label="Target")
    ax.fill_between(target_hist.bin_edges[:-1], target_hist.hist, alpha=0.5)
    ax.set_title(f"EMD: {emd:.4f}")
    
def temp_figs():
    checkpoints = {
        "Neighbor MoE": "/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling64_48105352/checkpoints/inference_rollouts/test_results.pt",
        "Neighbor MLP": "/pub/afeeney/bubbleformer_logs/neighbor_vit_poolboiling64_48103654/checkpoints/inference_rollouts/test_results.pt",
        #"Axial MoE": "/pub/afeeney/bubbleformer_logs/axial_moe_poolboiling64_48103671/checkpoints/inference_rollouts/test_results_clip_boiling_point.pt",
        #"Axial MLP": "/pub/afeeney/bubbleformer_logs/axial_vit_poolboiling64_48103668/checkpoints/inference_rollouts/test_results_clip_boiling_point.pt",
        #"Global MoE": "/pub/afeeney/bubbleformer_logs/vit_moe_poolboiling64_48103688/checkpoints/inference_rollouts/test_results_clip_boiling_point.pt",
        #"Global MLP": "/pub/afeeney/bubbleformer_logs/vit_poolboiling64_48103690/checkpoints/inference_rollouts/test_results_clip_boiling_point.pt",
    }

    fig, ax = plt.subplots(2 * len(checkpoints), 6, figsize=(10, 5), layout="constrained")
    for i, (name, checkpoint) in enumerate(checkpoints.items()):
        print("processing", name)
        test_results = torch.load(checkpoint, weights_only=False)
        #test_results = filter_sat(test_results)
        
        sub = [test_result for test_result in test_results if "subcooled" in test_result.case_name]
        print(len(sub), [s.case_name for s in sub])
        for j, test_result in enumerate(sub):
            
            plot_temp_dist(
                ax[i, j],
                test_result.preds[:, :100, :, 6:],
                test_result.targets[:, :100, :, 6:],
                test_result.fluid_params["bulk_temp"], 
                test_result.fluid_params["heater"]["wallTemp"]
            )
            
            plot_vel_dist(
                ax[len(checkpoints) + i, j],
                test_result.preds[:, :100, :, 6:], 
                test_result.targets[:, :100, :, 6:],
            )
            ax[-1, j].set_xlabel(pretty_name(test_result.case_name))
            
    ax[0, 0].set_ylabel("MoE T Dens.")
    ax[1, 0].set_ylabel("MLP T Dens.")
    ax[2, 0].set_ylabel("MoE Vel Dens.")
    ax[3, 0].set_ylabel("MLP Vel Dens.")
    ax[0, -1].legend()
    plt.savefig("sub_dist.pdf", bbox_inches="tight")
    plt.close()

temp_figs()
