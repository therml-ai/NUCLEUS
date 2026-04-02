import glob
from nucleus.plot.plotting import plot_temp
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

attention_probs_files = glob.glob("attention_probs_*.pt")

inp_file = glob.glob("inp_*.pt")[0]
inp = torch.load(inp_file)

for file in attention_probs_files[-5:-4]:
    print(file)
    attention_probs = torch.load(file)
    heads = attention_probs.shape[1]
    fig, axs = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    
    head_sum = attention_probs.sum(dim=1)
    norm = LogNorm(vmin=1e-2, vmax=1)
    plt.imshow(head_sum[0].to(torch.float32).cpu().numpy(), norm=norm)
    plt.xlabel("Linearized Key Index")
    plt.ylabel("Linearized Query Index")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label="Attention Probs Sum", fraction=0.04, pad=0.05, ticks=[1e-2, 1e-1, 1])
    plt.savefig(file.replace(".pt", "_head_sum.pdf"), bbox_inches="tight")
    plt.close()
    
for file in attention_probs_files[-5:-4]:
    print(file)
    attention_probs = torch.load(file)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    
    temp = torch.flipud(inp[0, 0, 1].to(torch.float32))
    im_temp = plot_temp(ax, temp, 58, 91)
    
    head_sum = attention_probs.sum(dim=1)

    for query_idx in [70, 200]:
        row, col = query_idx // 16, query_idx % 16
        query_probs = torch.nn.functional.interpolate(
            head_sum[0, query_idx].view(1, 1, 16, 16), 
            size=(64, 64), 
            mode="nearest"
        ).squeeze()
        query_probs = query_probs.to(torch.float32)
        query_probs = torch.flipud(query_probs).detach().cpu().numpy()
        query_probs[query_probs < 5e-2] = np.nan
        im_overlay = ax.imshow(
            query_probs, 
            cmap="binary",
            alpha=0.7,
            vmin=5e-2, 
            vmax=attention_probs[0].max().item()
        )

    
    plt.colorbar(im_temp, ax=ax, fraction=0.04, pad=0.05, ticks=[58, 91], label="Temperature", location="right")
    plt.colorbar(im_overlay, ax=ax, fraction=0.04, pad=0.05, ticks=[5e-2, 0.3, 0.7], label="Attention Score", location="bottom")
        
    plt.savefig("attention_probs_query_overlay.pdf", bbox_inches="tight")
    plt.close()