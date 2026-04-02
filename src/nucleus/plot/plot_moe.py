from termios import PARODD
import torch
from nucleus.test import TestResults
from nucleus.plot.plotting import ax_default, temp_cmap
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pathlib
from typing import List
import pandas as pd

def plot_rollout_moe_overlay(
    filename: str,
    rollout: torch.Tensor,
    test_results: TestResults,
):
    def make_plot(rollout):
        # moe_outputs is a List[List[TopkMoEOutput]], since for each timestep we store the moe outputs for every layer.
        moe_outputs_layer_0 = [m[0] for m in test_results.moe_outputs]
        
        num_experts = moe_outputs_layer_0[0].num_experts
        
        frames = 4
        fig, axs = plt.subplots(frames, num_experts, figsize=(10, 5), layout="constrained")
        for idx in range(frames):
            timestep = idx * 25
            print("plotting timestep", timestep)
        
            temp = torch.flipud(rollout[timestep, 1, :, :])
            
            # Model processes five frames at a time, so divide by 5 to get correct index into model output.
            moe_output = moe_outputs_layer_0[timestep // 5]
            assert moe_output.topk_indices.dim() == 5, "Topk indices must be of shape (1, T, H, W, topk)"
            assert moe_output.topk_indices.shape[0] == 1, "Topk indices must be of shape (1, T, H, W, topk)"
            topk_indices = moe_output.topk_indices.squeeze(0)[timestep % 5] # get the correct timestep from the model output.
            
            for expert_id, ax in enumerate(axs[idx]):
                im_temp = ax.imshow(temp, cmap=temp_cmap())
                
                expert_map = (topk_indices == expert_id).sum(dim=-1).float()
                expert_overlay = torch.nn.functional.interpolate(expert_map.view(1, 1, 16, 16), size=(64, 64), mode="nearest").squeeze()
                expert_overlay = torch.flipud(expert_overlay.bool())
                
                ax.imshow(expert_overlay, cmap="gray", vmin=0, vmax=1, alpha=0.7 * (expert_overlay > 0))
                ax.set_xticks([])
                ax.set_yticks([])
                
        for expert_id in range(num_experts):
            axs[0, expert_id].set_title(f"Expert {expert_id + 1}")
        for idx in range(frames):
            axs[idx, 0].set_ylabel(f"Step {idx * 25}")
        plt.colorbar(im_temp, ax=axs[:, -1].tolist())#, fraction=0.04, pad=0.05)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        
    make_plot(rollout.squeeze(0))

def pretty_name(case_name: str):
    if "_" in case_name:
        parts = case_name.split("_")
    else:
        parts = case_name.split(" ")
    parts[0] = parts[0].capitalize()
    parts[1] = parts[1].upper()
    return " ".join(parts)

def plot_routing_bar_chart(
    filename: str,
    test_results: List[TestResults]
):
    num_experts = test_results.moe_outputs[0][0].num_experts
    data = []
    for r in test_results:
        tokens_per_expert = torch.sum(torch.stack([t.tokens_per_expert for t in r.moe_outputs], dim=0), dim=0)
        routing_pct = tokens_per_expert / tokens_per_expert.sum()
        data[r.case_name] = routing_pct.tolist()
    
    df = pd.DataFrame(data)
    df = df.rename(columns=lambda x: " ".join(x.split("_")))
    df = df.rename(index=lambda x: f"Expert {x + 1}")
    df.T.plot(kind="bar", figsize=(12, 6))
    plt.xticks([pretty_name(p) for p in df.columns.tolist()], rotation=45, ha="right")
    plt.savefig(filename, bbox_inches="tight")

def routing_pct(tokens_per_expert: List[torch.Tensor]):
    r""" Gets the routing percentage for each expert across all timesteps.
    """
    assert tokens_per_expert.dim() == 1, "Tokens per expert must be of shape (num_experts,)"
    assert tokens_per_expert.sum() > 0
    timesteps = torch.stack(tokens_per_expert, dim=0)
    return timesteps.sum(dim=0) / timesteps.sum()

def plot_routing_heatmap(
    filename: str,
    test_results: List[TestResults]
):
    num_experts = test_results[0].moe_outputs[0][0].num_experts
    cases = [
        "saturated_fc72_104.0",
        "subcooled_fc72_97.0",
        "saturated_ln2_-176.0",
        "subcooled_ln2_-180.0"
    ]
    fig, axs = plt.subplots(1, len(cases), figsize=(12, 6), layout="constrained")
    
    r = [r for r in test_results if r.case_name in cases]
    
    routing_pcts = {}
    
    for idx, test_result in enumerate(r):
        tokens_per_expert = torch.stack([
            torch.stack([test_result.moe_outputs[timestep][layer].tokens_per_expert for layer in range(len(test_result.moe_outputs[0]))], dim=0)
            for timestep in range(len(test_result.moe_outputs))
        ], dim=0)
        
        print(tokens_per_expert.shape)
        
        # get the routing percentage of each expert across all timesteps.
        # Note that this does not combine different layers (dim=1).
        # [T, L, E] -> [L, E]
        routing_pct = tokens_per_expert.sum(dim=0) / tokens_per_expert.sum(dim=(0, 2), keepdim=True)[0]
        assert torch.allclose(routing_pct.sum(dim=-1), torch.ones(routing_pct.shape[0]))
        routing_pcts[test_result.case_name] = routing_pct
        
    max_pct = max([max(routing_pct.flatten()) for routing_pct in routing_pcts.values()])
        
    for idx, (case_name, routing_pct) in enumerate(routing_pcts.items()):
        im = axs[idx].imshow(routing_pct, cmap="rocket", vmin=0, vmax=max_pct)
        axs[idx].set_xlabel("Expert")
        axs[idx].set_ylabel("Layer")
        axs[idx].set_xticks(range(num_experts), labels=range(1, num_experts + 1))
        axs[idx].set_yticks(range(len(test_results[0].moe_outputs[0])), labels=range(1, len(test_results[0].moe_outputs[0]) + 1))
        axs[idx].set_title(pretty_name(case_name))
    plt.colorbar(im, ax=axs[-1], label="Routing Percentage", cmap="rocket", pad=0.05, fraction=0.04)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    
if __name__ == "__main__":
    
    root_dir = pathlib.Path("/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling64_48037580/checkpoints/inference_rollouts")
    test_results = torch.load(root_dir / "test_results.pt", weights_only=False)

    for test_result in test_results:
        save_filename = root_dir / f"moe_{test_result.case_name}.pdf"
        plot_rollout_moe_overlay(
            filename=save_filename,
             rollout=test_result.preds,
            test_results=test_result,
       )
    #plot_routing_bar_chart(
    #    filename=root_dir / "routing.pdf",
    #    test_results=test_results
    #)
    #plot_routing_heatmap(
    #    filename=root_dir / "routing_heatmap.pdf",
    #    test_results=test_results
    #)