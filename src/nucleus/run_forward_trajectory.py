from dataclasses import dataclass
from typing import Dict, Sequence, Union
import h5py
import json
import torch

from nucleus.trajectory import RolloutWindow, Trajectory

@dataclass
class TestResults:
    case_name: str
    # whatever the model's rollout returns: a cell-centered (T, H, W, 4) tensor, or a
    # natural-grid Trajectory for models that predict velocities on staggered faces
    preds: Union[torch.Tensor, Trajectory]
    targets: Union[torch.Tensor, Trajectory]
    sim_params: dict

def read_rollout_fields(test_file_path: str, field_names: Sequence[str]) -> Dict[str, torch.Tensor]:
    with h5py.File(test_file_path, "r") as handle:
        return {name: torch.from_numpy(handle[name][:]) for name in field_names}

def read_sim_params(test_file_path: str) -> dict:
    with open(test_file_path.replace(".hdf5", ".json"), "r") as handle:
        return json.load(handle)

def case_name(sim_params_dict: dict) -> str:
    return f"{sim_params_dict['setup']}_{sim_params_dict['liquid']}_{sim_params_dict['heater']['wallTemp']}"

def run_test(cfg, model, normalizer, test_file_path: str, trajectory_steps: int):
    if not hasattr(model, "rollout"):
        raise TypeError(f"{type(model).__name__} does not implement rollout()")
    fields = read_rollout_fields(test_file_path, model.rollout_fields)
    sim_params_dict = read_sim_params(test_file_path)
    window = RolloutWindow(
        start_time=cfg.start_time,
        history_steps=cfg.history_time_window,
        trajectory_steps=trajectory_steps,
    )
    with torch.inference_mode():
        preds, targets = model.rollout(fields, sim_params_dict, normalizer, window)
    return TestResults(case_name(sim_params_dict), preds, targets, sim_params=sim_params_dict)
