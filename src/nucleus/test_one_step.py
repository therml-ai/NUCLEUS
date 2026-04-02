from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
import numpy as np
from nucleus.data.batching import CollatedBatch, collate
from nucleus.data import BubbleForecast
from nucleus.layers.moe.topk_moe import TopkMoEOutput
from nucleus.utils.physical_metrics import PhysicalMetrics, BubbleMetrics, physical_metrics, bubble_metrics
from nucleus.utils.sdf_reinit import sdf_reinit
from nucleus.utils.normalize import normalize, unnormalize
import pathlib

@dataclass
class TestResults:
    case_name: str
    mae: float
    sdf_mae: float
    temp_mae: float
    velx_mae: float
    vely_mae: float
    
    def __repr__(self):
        return f"""
    Case: {self.case_name},
    MAE: {self.mae:.4f}, 
    SDF MAE: {self.sdf_mae:.4f},
    Temp MAE: {self.temp_mae:.4f}, 
    Velx MAE: {self.velx_mae:.4f}, 
    Vely MAE: {self.vely_mae:.4f}
"""

def run_test_one_step(model, cfg):
    
    results = []
    for test_path in cfg.data_cfg.test_paths:
        print(f"Testing {test_path}")
    
        downsample_factor = 1
        test_dataset = BubbleForecast(
            filenames=[test_path],
            input_fields=["dfun", "temperature", "velx", "vely"],
            output_fields=["dfun", "temperature", "velx", "vely"],
            norm="none",
            downsample_factor=downsample_factor,
            future_time_window=5,
            history_time_window=5,
            time_step=1,
            start_time=400,
            return_fluid_params=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=1,
            collate_fn=collate,
        )

        batch_sdf_maes = []
        batch_temp_maes = []
        batch_velx_maes = []
        batch_vely_maes = []
        with torch.inference_mode():
            for batch in test_dataloader:
                batch = batch.to("cuda")
                pred = model(batch.get_input())
                if isinstance(pred, tuple):
                    pred = pred[0]
                sdf_mae = torch.mean(torch.abs(pred[:, :, 0, :, :] - batch.target[:, :, 0, :, :]), dim=(1, 2, 3))
                temp_mae = torch.mean(torch.abs(pred[:, :, 1, :, :] - batch.target[:, :, 1, :, :]), dim=(1, 2, 3))
                velx_mae = torch.mean(torch.abs(pred[:, :, 2, :, :] - batch.target[:, :, 2, :, :]), dim=(1, 2, 3))
                vely_mae = torch.mean(torch.abs(pred[:, :, 3, :, :] - batch.target[:, :, 3, :, :]), dim=(1, 2, 3))
                batch_sdf_maes.extend(sdf_mae.tolist())
                batch_temp_maes.extend(temp_mae.tolist())
                batch_velx_maes.extend(velx_mae.tolist())
                batch_vely_maes.extend(vely_mae.tolist())
        
        fluid_params = batch.fluid_params_dict[0]
        case_name = fluid_params["setup"] + "_" + fluid_params["liquid"] + "_" + str(fluid_params["heater"]["wallTemp"])
        sdf_mae = np.mean(batch_sdf_maes)
        temp_mae = np.mean(batch_temp_maes)
        velx_mae = np.mean(batch_velx_maes)
        vely_mae = np.mean(batch_vely_maes)
        mae = np.mean(np.array(batch_sdf_maes) + np.array(batch_temp_maes) + np.array(batch_velx_maes) + np.array(batch_vely_maes))
        results.append(TestResults(
            case_name=case_name,
            mae=mae,
            sdf_mae=sdf_mae,
            temp_mae=temp_mae,
            velx_mae=velx_mae,
            vely_mae=vely_mae,
        ))
        print(results[-1])
    return results