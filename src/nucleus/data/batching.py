import dataclasses
import torch
from typing import Dict, List, Optional
from nucleus.data.normalize import Normalizer

@dataclasses.dataclass
class Data:
    input: torch.Tensor
    target: torch.Tensor
    sim_params_dict: Dict
    x_grid: torch.Tensor
    y_grid: torch.Tensor
    dx: float
    dy: float
    rollout_steps: Optional[int] = None
    sim_params_tensor: Optional[torch.Tensor] = None

    def to_collated_batch(self):
        return collate([self])

@dataclasses.dataclass
class CollatedBatch:
    input: torch.Tensor
    target: Optional[torch.Tensor]
    sim_params_dict: List[Dict]
    x_grid: torch.Tensor
    y_grid: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    rollout_steps: Optional[torch.Tensor] = None
    sim_params_tensor: Optional[torch.Tensor] = None

    def pin_memory(self):
        return CollatedBatch(
            input=self.input.pin_memory(),
            target=self.target.pin_memory() if self.target is not None else None,
            sim_params_dict=self.sim_params_dict,
            x_grid=self.x_grid.pin_memory(),
            y_grid=self.y_grid.pin_memory(),
            dx=self.dx.pin_memory(),
            dy=self.dy.pin_memory(),
            rollout_steps=self.rollout_steps,
            sim_params_tensor=self.sim_params_tensor.pin_memory() if self.sim_params_tensor is not None else None,
        )

    def to(self, device: torch.device, non_blocking: bool = False):
        return CollatedBatch(
            input=self.input.to(device, non_blocking=non_blocking),
            target=self.target.to(device, non_blocking=non_blocking) if self.target is not None else None,
            sim_params_dict=self.sim_params_dict,
            x_grid=self.x_grid.to(device, non_blocking=non_blocking),
            y_grid=self.y_grid.to(device, non_blocking=non_blocking),
            dx=self.dx.to(device, non_blocking=non_blocking),
            dy=self.dy.to(device, non_blocking=non_blocking),
            rollout_steps=self.rollout_steps,
            sim_params_tensor=self.sim_params_tensor.to(device, non_blocking=non_blocking) if self.sim_params_tensor is not None else None,
        )

    def detach(self):
        return CollatedBatch(
            input=self.input.detach(),
            target=self.target.detach() if self.target is not None else None,
            sim_params_dict=self.sim_params_dict,
            x_grid=self.x_grid.detach(),
            y_grid=self.y_grid.detach(),
            dx=self.dx.detach(),
            dy=self.dy.detach(),
            rollout_steps=self.rollout_steps,
            sim_params_tensor=self.sim_params_tensor.detach() if self.sim_params_tensor is not None else None,
        )

    def get_input(self):
        r"""
        This returns a copy of self, but without the target data,
        so this can be directly passed into the model as an input.
        """
        return CollatedBatch(
            input=self.input,
            target=None,
            sim_params_dict=self.sim_params_dict,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps,
            sim_params_tensor=self.sim_params_tensor
        )

    def fliplr(self):
        return CollatedBatch(
            # B T H W C, flip along the width (dim)
            input=torch.flip(self.input, dims=[3]),
            target=torch.flip(self.target, dims=[3]),
            sim_params_dict=self.sim_params_dict,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps,
            sim_params_tensor=self.sim_params_tensor
        )

    def noise_(self, scale):
        with torch.no_grad():
            self.input += torch.normal(0, scale, self.input.shape, device=self.input.device)

    def normalize(self, normalizer: Normalizer):
        return CollatedBatch(
            input=normalizer.normalize(self.input, self.get_temps()[0]),
            target=normalizer.normalize(self.target, self.get_temps()[0]),
            sim_params_dict=normalizer.normalize_params(self.sim_params_dict),
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps,
            sim_params_tensor=self.sim_params_tensor
        )

    def unnormalize(self, normalizer: Normalizer):
        return CollatedBatch(
            input=normalizer.unnormalize(self.input, self.get_temps()[0]),
            target=normalizer.unnormalize(self.target, self.get_temps()[0]),
            sim_params_dict=normalizer.unnormalize_params(self.sim_params_dict),
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps,
            sim_params_tensor=self.sim_params_tensor
        )

    def get_temps(self):
        bulk_temp = torch.tensor([d["bulk_temp"] for d in self.sim_params_dict], device=self.input.device)
        heater_temp = torch.tensor([d["heater"]["wallTemp"] for d in self.sim_params_dict], device=self.input.device)
        return bulk_temp, heater_temp

    def get_sim_params_tensor(self, device):
        assert self.sim_params_tensor is not None
        return self.sim_params_tensor.to(device)

def get_sim_params(sim_params_dict: Dict, fluid_params: List[str], heater_params: List[str], global_params: List[str]):
    fp = [sim_params_dict[param] for param in fluid_params]
    hp = [sim_params_dict["heater"][param] for param in heater_params]
    gp = [sim_params_dict[param] for param in global_params]
    return fp + hp + gp

def build_sim_params_tensor(sim_params_dict: Dict, fluid_params: List[str], heater_params: List[str], global_params: List[str]):
    return torch.tensor(get_sim_params(sim_params_dict, fluid_params, heater_params, global_params), dtype=torch.float32)

def make_data(
    input,
    target,
    sim_params_dict: dict,
    downsample_factor: int,
    fluid_params: List[str],
    heater_params: List[str],
    global_params: List[str]
):
    dx = (sim_params_dict["x_max"] - sim_params_dict["x_min"]) / (sim_params_dict["num_blocks_x"] * int(sim_params_dict["nx_block"]))
    dy = (sim_params_dict["y_max"] - sim_params_dict["y_min"]) / (sim_params_dict["num_blocks_y"] * int(sim_params_dict["ny_block"]))

    if downsample_factor > 1:
        dx *= downsample_factor
        dy *= downsample_factor

    # + dx / 2 since we're using a cell-centered grid.
    x_grid = torch.arange(sim_params_dict["x_min"], sim_params_dict["x_max"], dx) + dx / 2
    y_grid = torch.arange(sim_params_dict["y_min"], sim_params_dict["y_max"], dy) + dy / 2

    return Data(
        input=input,
        target=target,
        sim_params_dict=sim_params_dict,
        x_grid=x_grid,
        y_grid=y_grid,
        dx=dx,
        dy=dy,
        sim_params_tensor=build_sim_params_tensor(sim_params_dict, fluid_params, heater_params, global_params)
    )

def collate(data: List[Data]):
    return CollatedBatch(
        input=torch.stack([d.input for d in data]),
        target=torch.stack([d.target for d in data]),
        sim_params_dict=[d.sim_params_dict for d in data],
        x_grid=torch.stack([d.x_grid for d in data]),
        y_grid=torch.stack([d.y_grid for d in data]),
        dx=torch.tensor([d.dx for d in data]),
        dy=torch.tensor([d.dy for d in data]),
        sim_params_tensor=torch.stack([d.sim_params_tensor for d in data], dim=0),
    )