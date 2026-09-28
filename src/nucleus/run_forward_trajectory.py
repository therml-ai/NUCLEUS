from dataclasses import dataclass
from typing import Union
import h5py
import json
import torch

from nucleus.data.layout import convert_layout
from nucleus.trajectory import Trajectory
from nucleus.models.nucleus2_moe_divfree import Nucleus2MoEDivFree

@dataclass
class TestResults:
    case_name: str
    # A cell-centered (T, H, W, 4) tensor for the collocated models, or a natural-grid
    # Trajectory for the divfree model (velocities on their staggered faces).
    preds: Union[torch.Tensor, Trajectory]
    targets: Union[torch.Tensor, Trajectory]
    sim_params: dict

def _truncate_trajectory(trajectory: Trajectory, num_steps: int) -> Trajectory:
    # forward_trajectory rolls in output-window blocks, so it can overshoot
    # trajectory_steps; keep the first num_steps frames, which align with the ground
    # truth window (the initial frames are the true input, the rest are predictions).
    return Trajectory(
        sdf=trajectory.sdf[:, :num_steps],
        temp=trajectory.temp[:, :num_steps],
        velx=trajectory.velx[:, :num_steps],
        vely=trajectory.vely[:, :num_steps],
        sim_params=trajectory.sim_params,
    )

def run_test(cfg, model, normalizer, test_file_path: str, trajectory_steps: int):
    is_divfree = isinstance(model, Nucleus2MoEDivFree)
    with h5py.File(test_file_path, "r") as handle:
        sdf = torch.from_numpy(handle["dfun"][:])
        temp = torch.from_numpy(handle["temperature"][:])
        velx = torch.from_numpy(handle["velx"][:])
        vely = torch.from_numpy(handle["vely"][:])
        gt_trajectory = torch.stack((sdf, temp, velx, vely), dim=-1)
        if is_divfree:
            # The divfree model consumes the true staggered face velocities.
            velfacex = torch.from_numpy(handle["velfacex"][:])
            velfacey = torch.from_numpy(handle["velfacey"][:])

    json_path = test_file_path.replace(".hdf5", ".json")
    with open(json_path, "r") as handle:
        sim_params_dict: dict = json.load(handle)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    window = slice(cfg.start_time, cfg.start_time + cfg.history_time_window)

    with torch.inference_mode():
        if is_divfree:
            # Build the input Trajectory directly on the natural grids: sdf/temp
            # cell-centered, velx/vely on the staggered faces read from the file.
            initial_trajectory = Trajectory(
                sdf=sdf[window][None].to(device),
                temp=temp[window][None].to(device),
                velx=velfacex[window][None].to(device),
                vely=velfacey[window][None].to(device),
                sim_params=[sim_params_dict],
            )
            # The rolled prediction stays on its natural grids (velx/vely on faces) --
            # it is not averaged back to cell centers. The ground truth is built the
            # same way for a like-for-like comparison.
            pred_trajectory: Trajectory = model.forward_trajectory(
                initial_trajectory,
                normalizer,
                dx=1/32,
                input_time_window_size=8,
                output_time_window_size=8,
                trajectory_steps=trajectory_steps,
                use_sdf_reinit=False,
                clip_temp=False,
                use_leray=True,
                use_div_gate=True,
            )
            pred_trajectory = _truncate_trajectory(pred_trajectory, trajectory_steps)
            target_window = slice(cfg.start_time, cfg.start_time + trajectory_steps)
            targets = Trajectory(
                sdf=sdf[target_window][None],
                temp=temp[target_window][None],
                velx=velfacex[target_window][None],
                vely=velfacey[target_window][None],
                sim_params=[sim_params_dict],
            )
        else:
            initial_state = gt_trajectory[window][None, :].to(device)
            initial_cells = convert_layout(initial_state, target_layout=model.layout, source_layout="t h w c")
            pred_trajectory: torch.Tensor = model.forward_trajectory(
                initial_cells,
                sim_params_dict,
                normalizer,
                dx=1/32,
                input_time_window_size=8,
                output_time_window_size=8,
                trajectory_steps=trajectory_steps,
                use_sdf_reinit=False,
                return_moe_outputs=False,
                clip_temp=False
            )
            pred_trajectory = convert_layout(pred_trajectory, target_layout="t h w c", source_layout=model.layout).squeeze(0)
            targets = gt_trajectory[cfg.start_time : cfg.start_time + trajectory_steps]

    case_name = f"{sim_params_dict['setup']}_{sim_params_dict['liquid']}_{sim_params_dict['heater']['wallTemp']}"

    return TestResults(
        case_name,
        pred_trajectory,
        targets,
        sim_params=sim_params_dict
    )