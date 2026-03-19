import dataclasses
from this import d
import torch
from typing import Dict, List, Optional
from bubbleformer.data.normalize import Normalizer

@dataclasses.dataclass
class Data:
    input: torch.Tensor
    target: torch.Tensor
    fluid_params_dict: Dict
    x_grid: torch.Tensor
    y_grid: torch.Tensor
    dx: float
    dy: float
    rollout_steps: Optional[int] = None
    
    def to_collated_batch(self):
        return CollatedBatch(
            input=self.input.unsqueeze(0),
            target=self.target.unsqueeze(0) if self.target is not None else None,
            fluid_params_dict=[self.fluid_params_dict],
            x_grid=self.x_grid.unsqueeze(0),
            y_grid=self.y_grid.unsqueeze(0),
            dx=torch.tensor([self.dx]),
            dy=torch.tensor([self.dy]),
            rollout_steps=torch.tensor([self.rollout_steps]) if self.rollout_steps is not None else None
        )

@dataclasses.dataclass
class CollatedBatch:
    input: torch.Tensor
    target: Optional[torch.Tensor]
    fluid_params_dict: List[Dict]
    x_grid: torch.Tensor
    y_grid: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    rollout_steps: Optional[torch.Tensor] = None
    
    def to(self, device: torch.device):
        return CollatedBatch(
            input=self.input.to(device),
            target=self.target.to(device) if self.target is not None else None,
            fluid_params_dict=self.fluid_params_dict,
            x_grid=self.x_grid.to(device),
            y_grid=self.y_grid.to(device),
            dx=self.dx.to(device),
            dy=self.dy.to(device),
            rollout_steps=self.rollout_steps
        )
        
    def detach(self):
        return CollatedBatch(
            input=self.input.detach(),
            target=self.target.detach() if self.target is not None else None,
            fluid_params_dict=self.fluid_params_dict,
            x_grid=self.x_grid.detach(),
            y_grid=self.y_grid.detach(),
            dx=self.dx.detach(),
            dy=self.dy.detach(),
            rollout_steps=self.rollout_steps
        )
    
    def get_input(self):
        r"""
        This returns a copy of self, but without the target data,
        so this can be directly passed into the model as an input.
        """
        return CollatedBatch(
            input=self.input,
            target=None,
            fluid_params_dict=self.fluid_params_dict,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps
        )
        
    def fliplr(self):
        return CollatedBatch(
            # B T H W C, flip along the width (dim 3)
            input=torch.flip(self.input, dims=[3]),
            target=torch.flip(self.input, dims=[3]),
            fluid_params_dict=self.fluid_params_dict,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps
        )
        
    def noise(self, scale):
        noise = torch.normal(0, scale, self.input.shape, device=self.input.device)
        return CollatedBatch(
            input=self.input + noise,
            target=self.target,
            fluid_params_dict=self.fluid_params_dict,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps
        )
        
    def normalize(self, normalizer: Normalizer):
        return CollatedBatch(
            input=normalizer.normalize(self.input, self.get_temps()[0]),
            target=normalizer.normalize(self.target, self.get_temps()[0]),
            fluid_params_dict=self.fluid_params_dict,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
        )
        
    def unnormalize(self, normalizer: Normalizer):
        return CollatedBatch(
            input=normalizer.unnormalize(self.input, self.get_temps()[0]),
            target=normalizer.unnormalize(self.target, self.get_temps()[0]),
            fluid_params_dict=self.fluid_params_dict,
        )
        
    def gaussian_noise(self, sdf_scale: float, temp_scale: float, vel_scale: float):
        sdf_noise = torch.normal(0, sdf_scale, self.input[:, :, 0].shape, device=self.input.device)
        temp_noise = torch.normal(0, temp_scale, self.input[:, :, 1].shape, device=self.input.device)
        velx_noise = torch.normal(0, vel_scale, self.input[:, :, 2].shape, device=self.input.device)
        vely_noise = torch.normal(0, vel_scale, self.input[:, :, 3].shape, device=self.input.device)
        noisy_input = torch.stack([
            self.input[..., 0] + sdf_noise,
            self.input[..., 1] + temp_noise,
            self.input[..., 2] + velx_noise,
            self.input[..., 3] + vely_noise,
        ], dim=2)
        return CollatedBatch(
            input=noisy_input,
            target=self.target,
            fluid_params_dict=self.fluid_params_dict,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps
        )
        
    def get_temps(self):
        bulk_temp = torch.tensor([d["bulk_temp"] for d in self.fluid_params_dict], device=self.input.device)
        heater_temp = torch.tensor([d["heater"]["wallTemp"] for d in self.fluid_params_dict], device=self.input.device)
        return bulk_temp, heater_temp

    def normalize(self, normalizer: Normalizer):
        bulk_temp, heater_temp = self.get_temps()
        return CollatedBatch(
            input=normalizer.normalize(self.input, bulk_temp),
            target=normalizer.normalize(self.target, bulk_temp),
            fluid_params_dict=normalizer.normalize_params(self.fluid_params_dict),
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps
        )
    
    def unnormalize(self, normalizer: Normalizer):
        bulk_temp, heater_temp = self.get_temps()
        return CollatedBatch(
            input=normalizer.unnormalize(self.input, bulk_temp),
            target=normalizer.unnormalize(self.target, bulk_temp),
            fluid_params_dict=normalizer.unnormalize_params(self.fluid_params_dict),
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps
        )
        
    def fluid_params_tensor(self, device):
        return torch.tensor(
            [
                (
                    d["inv_reynolds"],
                    d["cpgas"],
                    d["mugas"],
                    d["rhogas"],
                    d["thcogas"],
                    d["stefan"],
                    d["prandtl"],
                    d["gravy"],
                    d["bulk_temp"],
                    d["heater"]["wallTemp"],
                    d["heater"]["nucWaitTime"],
                    d["heater"]["rcdAngle"],
                    d["heater"]["advAngle"],
                    d["heater"]["velContact"],
                    d["heater"]["xMin"],
                    d["heater"]["xMax"]
                ) for d in self.fluid_params_dict
            ],
            dtype=torch.float32,
            device=device
        )
    
def make_data(input, target, fluid_params_dict, downsample_factor: int, rollout_steps: Optional[int] = None):
    dx = (fluid_params_dict["x_max"] - fluid_params_dict["x_min"]) / (fluid_params_dict["num_blocks_x"] * int(fluid_params_dict["nx_block"]))
    dy = (fluid_params_dict["y_max"] - fluid_params_dict["y_min"]) / (fluid_params_dict["num_blocks_y"] * int(fluid_params_dict["ny_block"]))

    if downsample_factor > 1:
        dx *= downsample_factor
        dy *= downsample_factor

    # + dx / 2 since we're using a cell-centered grid.
    x_grid = torch.arange(fluid_params_dict["x_min"], fluid_params_dict["x_max"], dx) + dx / 2
    y_grid = torch.arange(fluid_params_dict["y_min"], fluid_params_dict["y_max"], dy) + dy / 2

    return Data(
        input=input,
        target=target,
        fluid_params_dict=fluid_params_dict,
        x_grid=x_grid,
        y_grid=y_grid,
        dx=dx,
        dy=dy,
        rollout_steps=rollout_steps
    )

def collate(data: List[Data]):    
    return CollatedBatch(
        input=torch.stack([d.input for d in data]),
        target=torch.stack([d.target for d in data]),
        fluid_params_dict=[d.fluid_params_dict for d in data],
        x_grid=torch.stack([d.x_grid for d in data]),
        y_grid=torch.stack([d.y_grid for d in data]),
        dx=torch.tensor([d.dx for d in data]),
        dy=torch.tensor([d.dy for d in data]),
    )