from dataclasses import dataclass
from typing import List, Tuple
import torch
import numpy as np
from nucleus.data.batching import Data
from nucleus.data import ForecastDataset, InMemForecastDataset
from nucleus.layers.moe.topk_moe import TopkMoEOutput
from nucleus.utils.physical_metrics import PhysicalMetrics, BubbleMetrics, physical_metrics, bubble_metrics
from nucleus.utils.sdf_reinit import sdf_reinit_fast_marching


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
    fluid_params: dict

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
    
def clip_liquid_temp(preds, fluid_params):
    liquid = fluid_params["liquid"]
    if liquid == "fc72":
        max_liquid_temp = 58
    elif liquid == "r515b":
        max_liquid_temp = -19
    elif liquid == "ln2":
        max_liquid_temp = -196
    sdf = preds[:, :, 0, :, :]
    temp = preds[:, :, 1, :, :]
    liquid_mask = sdf < 0
    temp[liquid_mask] = torch.clamp(temp[liquid_mask], max=max_liquid_temp, min=fluid_params["bulk_temp"])
    temp[~liquid_mask] = torch.clamp(temp[~liquid_mask], min=fluid_params["bulk_temp"])
    return temp
    
def run_test(model, normalizer, test_file_path: str, max_timesteps: int):
    test_dataset = InMemForecastDataset(
        filenames=[test_file_path],
        input_fields=["dfun", "temperature", "velx", "vely"],
        output_fields=["dfun", "temperature", "velx", "vely"],
        future_time_window=5,
        history_time_window=5,
        time_step=1,
        start_time=400,
        normalizer=normalizer,
        augment=False
    )

    start_time = test_dataset.start_time
    skip_itrs = test_dataset.future_time_window
    preds = []
    targets = []
    timesteps = []
    moe_outputs = []

    with torch.inference_mode():
        for itr in range(0, max_timesteps, skip_itrs):            
            data: Data = test_dataset[itr]
            
            batch = data.to_collated_batch()
            batch = batch.to("cuda")
            
            # these values were normalized when indexing the dataset. We need the unnormalized
            # values so we can properly normalize/unnormalize the data fields in the inference loop.
            bulk_temp = normalizer.unnormalize_params(batch.fluid_params_dict)[0]["bulk_temp"]
            wall_temp = normalizer.unnormalize_params(batch.fluid_params_dict)[0]["heater"]["wallTemp"]
            
            if len(preds) > 0:
                batch.input = preds[-1].unsqueeze(0).to(batch.input.device)
                batch.input = normalizer.normalize(batch.input, bulk_temp)
            tgt = batch.target

            torch.compiler.cudagraph_mark_step_begin()
            output = model(batch.get_input())
            if isinstance(output, tuple):
                pred, moe_output = output
            else:
                pred = output
                moe_output = []
            
            if len(moe_output) > 0:
                # NOTE: only tracking moe outputs for every layer. Must move to CPU to avoid mem overflow.
                moe_outputs.append([m.detach().to('cpu') for m in moe_output])

            # clip pred temperature to be below wall temperature.
            pred[..., 1] = torch.clamp(
                pred[..., 1], 
                max=wall_temp
            )
            
            pred = normalizer.unnormalize(pred, bulk_temp)
            tgt = normalizer.unnormalize(tgt, bulk_temp)

            pred = pred.to(torch.float32).squeeze(0).detach().cpu()
            tgt = tgt.to(torch.float32).squeeze(0).detach().cpu()

            # Reinitialize the SDF at each timestep
            #pred[:, 0] = sdf_reinit_fast_marching(pred[:, 0], dx=1 / 4, far_threshold=4)

            preds.append(pred)
            targets.append(tgt)
            timesteps.append(torch.arange(start_time+itr+skip_itrs, start_time+itr+2*skip_itrs))
            
            #torch.save(inp, f"inp_{itr}.pt")
            #break

    preds = torch.cat(preds, dim=0)[None, ...]         # 1, T, H, W, C
    targets = torch.cat(targets, dim=0)[None, ...]     # 1, T, H, W, C
    timesteps = torch.cat(timesteps, dim=0)             # T,
    
    fluid_params = test_dataset.fluid_params[0]

    print("-"*100)
    print(f"Rollout Statistics on {test_file_path}:")
    dx = 1/4
    dy = dx
    bulk_temp = fluid_params["bulk_temp"]
    heater_temp = fluid_params["heater"]["wallTemp"]
    pred_physical_metrics = physical_metrics(
        preds[..., 0], 
        preds[..., 1], 
        preds[..., 2], 
        preds[..., 3],
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
    pred_bubble_metrics = bubble_metrics(preds[..., 0], preds[..., 2], preds[..., 3], dx=dx, dy=dy)
    
    target_physical_metrics = physical_metrics(
        targets[..., 0], 
        targets[..., 1], 
        targets[..., 2], 
        targets[..., 3],
        # TODO: get these from dataset
        heater_min=-5.25,
        heater_max=5.25,
        bulk_temp=bulk_temp,
        heater_temp=heater_temp,
        xcoords=torch.arange(-8, 8, dx) + dx / 2,
        dx=dx,
        dy=dy
    )
    target_bubble_metrics = bubble_metrics(targets[..., 0], targets[..., 2], targets[..., 3], dx=dx, dy=dy)
    
    metric_distribution_pred = metric_distribution(pred_physical_metrics, pred_bubble_metrics)
    metric_distribution_target = metric_distribution(target_physical_metrics, target_bubble_metrics)
    
    case_name = f"{fluid_params['setup']}_{fluid_params['liquid']}_{fluid_params['heater']['wallTemp']}"
    
    print(f"{case_name}, Pred Metrics: ")
    print(metric_distribution_pred)
    print(f"{case_name}, Target Metrics: ")
    print(metric_distribution_target)
    
    return TestResults(
        case_name,
        preds, 
        targets, 
        pred_physical_metrics, 
        target_physical_metrics, 
        pred_bubble_metrics, 
        target_bubble_metrics, 
        moe_outputs, 
        # the test dataset is only one file, so we can take the first index in fluid_params
        fluid_params=fluid_params
    )