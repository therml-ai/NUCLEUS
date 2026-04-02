import torch
from nucleus.test import TestResults
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_simple_metrics(save_dir: str, test_results: TestResults):
    pred_metrics = pd.DataFrame({
        "Void Fraction": test_results.pred_physical_metrics.vapor_volume.squeeze(0),
        "Liquid Temperature": test_results.pred_physical_metrics.mean_liquid_temperature.squeeze(0),
        "Heater Temperature": test_results.pred_physical_metrics.liquid_temperature_at_heater.squeeze(0),
        "Liquid Velocity": test_results.pred_physical_metrics.mean_liquid_x_velocity.squeeze(0),
        "Vapor Velocity": test_results.pred_physical_metrics.mean_vapor_x_velocity.squeeze(0),
        "Eikonal Error": test_results.pred_physical_metrics.eikonal.squeeze(0)
    })
    target_metrics = pd.DataFrame({
        "Void Fraction": test_results.target_physical_metrics.vapor_volume.squeeze(0),
        "Liquid Temperature": test_results.target_physical_metrics.mean_liquid_temperature.squeeze(0),
        "Heater Temperature": test_results.target_physical_metrics.liquid_temperature_at_heater.squeeze(0),
        "Liquid Velocity": test_results.target_physical_metrics.mean_liquid_x_velocity.squeeze(0),
        "Vapor Velocity": test_results.target_physical_metrics.mean_vapor_x_velocity.squeeze(0),
        "Eikonal Error": test_results.target_physical_metrics.eikonal.squeeze(0)
    })
    
    print(pred_metrics)
    print(target_metrics)
    
    columns = pred_metrics.columns
    x = np.arange(len(columns))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.bar(x - width / 2, pred_metrics.mean(), yerr=pred_metrics.std(), width=width, color="blue", label="Predicted")
    ax.bar(x + width / 2, target_metrics.mean(), yerr=target_metrics.std(), width=width, color="red", label="Target")
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.legend()
    ax.set_ylabel("Time-averaged Value")
    plt.savefig(f"{save_dir}/simple_metrics.pdf", bbox_inches="tight")
    plt.close()

def plot_bubble_counts(save_dir: str, test_results: TestResults):
    pred_bubble_counts = test_results.pred_bubble_metrics.bubble_count.squeeze(0)
    target_bubble_counts = test_results.target_bubble_metrics.bubble_count.squeeze(0)
    plt.plot(torch.arange(len(pred_bubble_counts)), pred_bubble_counts, label="Predicted")
    plt.plot(torch.arange(len(target_bubble_counts)), target_bubble_counts, label="Target")
    plt.legend()
    plt.xlabel("Bubble Count")
    plt.ylabel("Count")
    plt.savefig(f"{save_dir}/bubble_counts.pdf", bbox_inches="tight")
    plt.close()

def plot_vapor_volume_at_height(save_dir: str, test_results: TestResults):
    r"""
    This plots the vapor volume at each height.
    """
    pred_vapor_volume_at_height = test_results.pred_physical_metrics.vapor_volume_at_height.squeeze(0)
    target_vapor_volume_at_height = test_results.target_physical_metrics.vapor_volume_at_height.squeeze(0)    
    plt.violinplot(
        # Time averaged vapor volume at each height
        [pred_vapor_volume_at_height.mean(dim=0), target_vapor_volume_at_height.mean(dim=0)],
        showmeans=True,
        showmedians=True,
    )
    plt.ylabel("Height")
    plt.savefig(f"{save_dir}/vapor_volume_at_height.pdf", bbox_inches="tight")
    plt.close()