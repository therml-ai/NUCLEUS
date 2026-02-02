from dataclasses import dataclass
from typing import List
import torch
from bubbleformer.data.batching import Data
from bubbleformer.data import BubbleForecast
from bubbleformer.layers.moe.topk_moe import TopkMoEOutput
from bubbleformer.utils.physical_metrics import PhysicalMetrics, BubbleMetrics, physical_metrics, bubble_metrics
from bubbleformer.utils.sdf_reinit import sdf_reinit
from bubbleformer.utils.normalize import normalize, unnormalize

@dataclass
class TestResults:
    preds: torch.Tensor
    targets: torch.Tensor
    p: PhysicalMetrics
    b: BubbleMetrics
    moe_outputs: List[TopkMoEOutput]
    fluid_params: dict

def run_test(model, test_file_path: str, max_timesteps: int):
    downsample_factor = 1
    test_dataset = BubbleForecast(
        filenames=[test_file_path],
        input_fields=["dfun", "temperature", "velx", "vely"],
        output_fields=["dfun", "temperature", "velx", "vely"],
        norm="none",
        downsample_factor=downsample_factor,
        time_window=5,
        start_time=200,
        return_fluid_params=True,
    )

    start_time = test_dataset.start_time
    skip_itrs = test_dataset.time_window
    preds = []
    targets = []
    timesteps = []
    moe_outputs = []

    with torch.inference_mode():
        for itr in range(0, max_timesteps, skip_itrs):
            print(f"Processing timestep {itr} / {max_timesteps}")
            
            data: Data = test_dataset[itr]
            
            batch = data.to_collated_batch()
            batch = batch.to("cuda")

            if len(preds) > 0:
                batch.input = preds[-1].unsqueeze(0).to(batch.input.device)
            tgt = batch.target

            output = model(batch.get_input())
            if isinstance(output, tuple):
                pred, moe_output = output
            else:
                pred = output
                moe_output = []
            
            # NOTE: only tracking moe outputs for the first layer
            # all tensor are moved to the CPU
            if len(moe_output) > 0:
                moe_outputs.append(moe_output[0].detach().to('cpu'))
                
            #pred = unnormalize(pred, data.fluid_params_dict["bulk_temp"], data.fluid_params_dict["heater"]["wallTemp"])
            #tgt = unnormalize(tgt, data.fluid_params_dict["bulk_temp"], data.fluid_params_dict["heater"]["wallTemp"])

            # clip pred temperature to valid range, between liquid bulk temp and heater temp.
            pred[:, :, 1] = torch.clamp(
                pred[:, :, 1], 
                min=data.fluid_params_dict["bulk_temp"], 
                max=data.fluid_params_dict["heater"]["wallTemp"]
            )
            
            pred = pred.to(torch.float32).squeeze(0).detach().cpu()
            tgt = tgt.to(torch.float32).squeeze(0).detach().cpu()
            
            print(data.dx)
            
            # Reinitialize the SDF at each timestep
            #pred[:, 0] = sdf_reinit(pred[:, 0], dx=data.dx, far_threshold=2)
            
            preds.append(pred)
            targets.append(tgt)
            timesteps.append(torch.arange(start_time+itr+skip_itrs, start_time+itr+2*skip_itrs))
            
            #torch.save(inp, f"inp_{itr}.pt")
            #break

    preds = torch.cat(preds, dim=0)[None, ...]         # 1, T, C, H, W
    targets = torch.cat(targets, dim=0)[None, ...]     # 1, T, C, H, W
    timesteps = torch.cat(timesteps, dim=0)             # T,

    topk_indices = [moe_output.topk_indices.squeeze(0) for moe_output in moe_outputs]
    if topk_indices:
        topk_indices = torch.cat(topk_indices, dim=0) # (T, H, W, topk)
    else:
        topk_indices = None

    print("-"*100)
    print(f"Rollout Statistics on {test_file_path}:")
    dx = 1/32 * downsample_factor
    dy = dx
    """
    p = physical_metrics(
        preds[:, :, 0], 
        preds[:, :, 1], 
        preds[:, :, 2], 
        preds[:, :, 3],
        # TODO: get these from dataset
        heater_min=-5.25,
        heater_max=5.25,
        # TODO: get from dataset
        xcoords=torch.arange(-8, 8, dx) + dx / 2,
        # TODO: get from dataset
        dx=dx, 
        dy=dy
    )
    b = bubble_metrics(preds[:, :, 0], preds[:, :, 2], preds[:, :, 3], dx=dx, dy=dy)
    """
    p = None
    b = None
    
    # NOTE: the test dataset is only one file, so we can take the first index in fluid_params
    return TestResults(preds, targets, p, b, moe_outputs, fluid_params=test_dataset.fluid_params[0])