from dataclasses import dataclass
import h5py
import json
import numpy as np
import torch
from typing import List, Tuple

from nucleus.data.batching import Data
from nucleus.data import ForecastDataset, InMemForecastDataset
from nucleus.data.layout import convert_layout
from nucleus.layers.moe.topk_moe import TopkMoEOutput
from nucleus.utils.physical_metrics import PhysicalMetrics, BubbleMetrics, physical_metrics, bubble_metrics
#from nucleus.baseline.poseidon import ScOTOutput
#from nucleus.baseline.moe_dpot import MoEPOTNet

@dataclass
class TestResults:
    case_name: str
    preds: torch.Tensor
    targets: torch.Tensor
    pred_physical_metrics: PhysicalMetrics
    target_physical_metrics: PhysicalMetrics
    pred_bubble_metrics: BubbleMetrics
    target_bubble_metrics: BubbleMetrics
    moe_outputs: List[List[TopkMoEOutput]] # Timesteps x Layers x TopkMoEOutput
    sim_params: dict

@dataclass
class TimeDistributionMetrics:
    vapor_volume: Tuple[float, float]
    heater_temp: Tuple[float, float]
    liquid_temp: Tuple[float, float]
    bubble_count: Tuple[float, float]
    bubble_volume: Tuple[float, float]
    bubble_x_velocity: Tuple[float, float]
    bubble_y_velocity: Tuple[float, float]
    abs_max_eikonal_error: float

def bubble_metric(bubble_metric: List[List[List[float]]]):
    bubbles = []
    for batch in bubble_metric:
        for timestep in batch:
            for bubble_metric in timestep:
                bubbles.append(bubble_metric)
    flat = np.array(bubbles)
    return np.mean(flat), np.std(flat)

def metric_distribution(physical_metrics: PhysicalMetrics, bubble_metrics: BubbleMetrics):
    vapor_volume = physical_metrics.vapor_volume
    # TODO: Do something else to compare temps..., not normally distributed
    liquid_temp = physical_metrics.mean_liquid_temperature
    heater_temp = physical_metrics.liquid_temperature_at_heater
    
    bubble_count = bubble_metrics.bubble_count
    bubble_volume = bubble_metrics.bubble_volume
    bubble_x_velocity = bubble_metrics.bubble_x_velocity
    bubble_y_velocity = bubble_metrics.bubble_y_velocity
    
    return TimeDistributionMetrics(
        vapor_volume=(vapor_volume.mean().item(), vapor_volume.std().item()),
        heater_temp=None,#(heater_temp.mean(dim=(1, 2)).item(), heater_temp.std(dim=(1, 2)).item()),
        liquid_temp=(liquid_temp.mean().item(), liquid_temp.std().item()),
        bubble_count=(bubble_count.float().mean().item(), bubble_count.float().std().item()),
        bubble_volume=bubble_metric(bubble_volume),
        bubble_x_velocity=bubble_metric(bubble_x_velocity),
        bubble_y_velocity=bubble_metric(bubble_y_velocity),
        abs_max_eikonal_error=(1 - physical_metrics.eikonal).abs().max().item()
    )
    
def run_test(cfg, model, normalizer, test_file_path: str, trajectory_steps: int):
    with h5py.File(test_file_path, "r") as handle:
        sdf = torch.from_numpy(handle["dfun"][:])
        temp = torch.from_numpy(handle["temperature"][:])
        velx = torch.from_numpy(handle["velx"][:])
        vely = torch.from_numpy(handle["vely"][:])
        gt_trajectory = torch.stack((sdf, temp, velx, vely), dim=-1)

    initial_state: torch.Tensor = gt_trajectory[:cfg.history_time_window][None, :]
    json_path = test_file_path.replace(".hdf5", ".json")
    with open(json_path, "r") as handle:
        sim_params_dict: dict = json.load(handle)
    sim_params_tensor = torch.Tensor(
        [sim_params_dict[param] for param in model.expected_fluid_params] +
        [sim_params_dict["heater"][param] for param in model.expected_heater_params] +
        [sim_params_dict[param] for param in model.expected_global_params],
    )[None, :]
    
    normalized_initial_state = normalizer.normalize(initial_state, bulk_temp=sim_params_dict["bulk_temp"])
    normalized_sim_params_tensor = normalizer.normalize_params(sim_params_tensor)

    print(normalized_initial_state.dtype)

    with torch.inference_mode():
        normalized_pred_trajectory: torch.Tensor = model.forward_trajectory(
            convert_layout(normalized_initial_state, target_layout=model.layout, source_layout="t h w c"),
            normalized_sim_params_tensor,
            dx=1/4,
            input_time_window_size=8,
            output_time_window_size=8,
            trajectory_steps=trajectory_steps,
            use_sdf_reinit=False,
            return_moe_outputs=False
        )
        
    pred_trajectory = normalizer.unnormalize(normalized_pred_trajectory, bulk_temp=sim_params_dict["bulk_temp"])
    pred_trajectory = convert_layout(pred_trajectory, target_layout="t h w c", source_layout=model.layout)
    pred_trajectory = pred_trajectory.squeeze(0)

    """
    print("-"*100)
    print(f"Rollout Statistics on {test_file_path}:")
    dx = 1/4
    dy = dx
    bulk_temp = sim_params_dict["bulk_temp"]
    heater_temp = sim_params_dict["heater"]["wallTemp"]
    pred_physical_metrics = physical_metrics(
        pred_trajectory[..., 0], 
        pred_trajectory[..., 1], 
        pred_trajectory[..., 2], 
        pred_trajectory[..., 3],
        # TODO: get these from dataset
        heater_min=-5.25,
        heater_max=5.25,
        bulk_temp=bulk_temp,
        heater_temp=heater_temp,
        # TODO: get from dataset
        xcoords=torch.arange(-8, 8, dx) + dx / 2,
        # TODO: get from dataset
        dx=dx, 
        dy=dy
    )
    pred_bubble_metrics = bubble_metrics(pred_trajectory[..., 0], pred_trajectory[..., 2], pred_trajectory[..., 3], dx=dx, dy=dy)
    
    target_physical_metrics = physical_metrics(
        gt_trajectory[..., 0], 
        gt_trajectory[..., 1], 
        gt_trajectory[..., 2], 
        gt_trajectory[..., 3],
        # TODO: get these from dataset
        heater_min=-5.25,
        heater_max=5.25,
        bulk_temp=bulk_temp,
        heater_temp=heater_temp,
        xcoords=torch.arange(-8, 8, dx) + dx / 2,
        dx=dx,
        dy=dy
    )
    target_bubble_metrics = bubble_metrics(gt_trajectory[..., 0], gt_trajectory[..., 2], gt_trajectory[..., 3], dx=dx, dy=dy)
    
    metric_distribution_pred = metric_distribution(pred_physical_metrics, pred_bubble_metrics)
    metric_distribution_target = metric_distribution(target_physical_metrics, target_bubble_metrics)
    
    
    print(f"{case_name}, Pred Metrics: ")
    print(metric_distribution_pred)
    print(f"{case_name}, Target Metrics: ")
    print(metric_distribution_target)
    """
    
    case_name = f"{sim_params_dict['setup']}_{sim_params_dict['liquid']}_{sim_params_dict['heater']['wallTemp']}"

    return TestResults(
        case_name,
        pred_trajectory, 
        gt_trajectory, 
        None, 
        None, 
        None, 
        None, 
        None, 
        sim_params=sim_params_dict
    )