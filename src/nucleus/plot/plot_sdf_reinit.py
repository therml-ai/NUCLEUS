import torch
import matplotlib.pyplot as plt
from nucleus.plot.plotting import plot_sdf
from nucleus.utils.physical_metrics import eikonal

def plot_sdf_reinit(sat_pred, sub_pred, sat_pred_with_reinit, sub_pred_with_reinit):
    fig, axs = plt.subplots(2, 2, figsize=(6, 5), layout="constrained")
    plot_sdf(axs[0, 0], torch.flipud(sat_pred))
    plot_sdf(axs[0, 1], torch.flipud(sub_pred))
    plot_sdf(axs[1, 0], torch.flipud(sat_pred_with_reinit))
    im = plot_sdf(axs[1, 1], torch.flipud(sub_pred_with_reinit))
    
    axs[0, 0].set_title("Saturated SDF FC72 91 °C")
    axs[0, 1].set_title("Subcooled SDF FC72 97 °C")
    
    axs[0, 0].set_ylabel(r"No SDF Reinit")
    axs[1, 0].set_ylabel(r"With SDF Reinit")
    
    axs[0, 0].set_xticks([])
    axs[0, 1].set_xticks([])
    axs[1, 0].set_xticks([])
    axs[1, 1].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 1].set_yticks([])
    axs[1, 0].set_yticks([])
    axs[1, 1].set_yticks([])
    
    plt.colorbar(im, ax=axs[:, 1], ticks=[-6, -4, -2, 0, 0.2], fraction=0.04, pad=0.05)
    plt.savefig("sdf_reinit_comparison.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    preds = torch.load("/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling64_48025794/checkpoints/inference_rollouts/test_results.pt", weights_only=False)
    preds_with_reinit = torch.load("/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling64_48025794/checkpoints/inference_rollouts/test_results_reinit.pt", weights_only=False)
    
    sat_pred = preds[0]
    sub_pred = preds[1]
    
    print(sat_pred.preds.shape)
    
    plot_sdf_reinit(
        sat_pred=preds[0].preds[0, 25, 0],
        sub_pred=preds[1].preds[0, 25, 0],
        sat_pred_with_reinit=preds_with_reinit[0].preds[0, 25, 0],
        sub_pred_with_reinit=preds_with_reinit[1].preds[0, 25, 0]
    )