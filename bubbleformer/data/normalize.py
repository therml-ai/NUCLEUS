import torch
import numpy as np
from dataclasses import dataclass
from omegaconf import DictConfig
import hydra
import math
from typing import Optional, Callable
import yaml

@dataclass
class NormalizerConstants:
    absmax_temp: float
    temp_half_std: float
    sdf_mean: float
    max_domain_size: float
    velx_std: float
    vely_std: float
    numeric_fluid_params_min: Optional[dict] = None
    numeric_fluid_params_max: Optional[dict] = None
    
    def to_yaml_string(self) -> str:
        r"""
        This returns a YAML string that can be used as a config file for the normalizer.
        """
        fluid_params_min_yaml = yaml.dump(self.numeric_fluid_params_min, default_flow_style=False) if self.numeric_fluid_params_min is not None else ""
        fluid_params_max_yaml = yaml.dump(self.numeric_fluid_params_max, default_flow_style=False) if self.numeric_fluid_params_max is not None else ""
        fluid_params_yaml = [
            f"absmax_temp: {self.absmax_temp}",
            f"max_temp_diff: {self.max_temp_diff}",
            f"max_domain_size: {self.max_domain_size}",
            f"velx_std: {self.velx_std}",
            f"vely_std: {self.vely_std}",
        ]
        
        fmin = yaml.dump({"fluid_params_min": self.numeric_fluid_params_min}, default_flow_style=False) if self.numeric_fluid_params_min is not None else None
        fmax = yaml.dump({"fluid_params_max": self.numeric_fluid_params_max}, default_flow_style=False) if self.numeric_fluid_params_max is not None else None
        
        if fmin:
            fluid_params_yaml.append(fmin)
        if fmax:
            fluid_params_yaml.append(fmax)
        
        return "\n".join(fluid_params_yaml)

def minmax_normalize(value: float, min: float, max: float) -> float:
    return (value - min) / (max - min)

def minmax_unnormalize(value: float, min: float, max: float) -> float:
    return value * (max - min) + min

class Normalizer:
    r"""
    This normalizes the sdf, temperature, and velocity fields. It also performs minmax normalization on the fluid parameters.
    1. sdf is normalized to [-1, 1] using the max training domain size
    2. temperature is normalized to [0, 1] using the bulk temperature and absolute maximum temperature across all training data.
       - the bulk temperature MUST be passed in for this to work.
    3. the velocities are normalized using the std across all training data.
    """
    def __init__(self, constants):
        self.constants = constants
        
    def normalize_params(self, fluid_params_dict: dict) -> dict:
        return dict([
            (key, minmax_normalize(value, self.constants.numeric_fluid_params_min[key], self.constants.numeric_fluid_params_max[key]))
            for key, value in fluid_params_dict.items()
        ])

    def unnormalize_params(self, fluid_params_dict: dict) -> dict:
        return dict([
            (key, minmax_unnormalize(value, self.constants.numeric_fluid_params_min[key], self.constants.numeric_fluid_params_max[key]))
            for key, value in fluid_params_dict.items()
        ])
        
    def normalize_temp(self, temp: torch.Tensor, bulk_temp: torch.Tensor) -> torch.Tensor:
        return torch.log1p(temp - bulk_temp)
    
    def unnormalize_temp(self, temp: torch.Tensor, bulk_temp: torch.Tensor) -> torch.Tensor:
        return torch.expm1(temp) + bulk_temp

    def normalize_velx(self, vel: torch.Tensor) -> torch.Tensor:
        return vel / self.constants.velx_std
    
    def unnormalize_velx(self, vel: torch.Tensor) -> torch.Tensor:
        return vel * self.constants.velx_std
    
    def normalize_vely(self, vel: torch.Tensor) -> torch.Tensor:
        return vel / self.constants.vely_std
    
    def unnormalize_vely(self, vel: torch.Tensor) -> torch.Tensor:
        return vel * self.constants.vely_std
    
    def normalize_sdf(self, sdf: torch.Tensor) -> torch.Tensor:
        return sdf / (self.constants.max_domain_size)
    
    def unnormalize_sdf(self, sdf: torch.Tensor) -> torch.Tensor:
        return sdf * (self.constants.max_domain_size)

class RunningVariance:
    def __init__(self, bins: int, range: tuple[float, float]):
        self.hist = np.array([0]*(bins - 1), dtype=np.int64)
        self.range = range
        self.bins = np.linspace(range[0], range[1], bins, dtype=np.float64)
        self.count = 0
        
    def update(self, value: np.ndarray):
        numel = value.size
        if numel == 0:
            return
        self.hist += np.histogram(value.astype(np.int64), bins=self.bins, range=self.range)[0]
        self.count += numel
    
    def var(self) -> float:
        bin_mid_points = (self.bins[1:] + self.bins[:-1]) / 2
        mean = np.average(bin_mid_points, weights=self.hist)
        return np.average((bin_mid_points - mean) ** 2, weights=self.hist).item()
    
    def std(self) -> float:
        return math.sqrt(self.var())
    
    def mean(self) -> float:
        bin_mid_points = (self.bins[1:] + self.bins[:-1]) / 2
        return np.average(bin_mid_points, weights=self.hist).item()
    
def is_number(value: any) -> bool:
    if isinstance(value, (float, int)):
        return True
    # If it's a string, try converting to float
    try:
        float(value)
        return True
    except:
        return False
    
def nested_dict_minmax(dict1: dict, dict2: dict, op: Callable) -> dict:
    r"""
    Applies a reduction operation `op` to two nested dictionaries (i.e., potentially a dict of dicts). This assumes that the
    dictionaries have identical structure. This only applies to numeric values, so strings are excluded from the output.
    """
    out_dict = {}
    for key in dict1.keys():
        if isinstance(dict1[key], dict):
            out_dict[key] = nested_dict_minmax(dict1[key], dict2[key], op)
        elif is_number(dict1[key]) and is_number(dict2[key]):
            out_dict[key] = op(dict1[key], dict2[key])
    return out_dict

def nested_dict_min(dict1: dict, dict2: dict) -> dict:
    return nested_dict_minmax(dict1, dict2, min)

def nested_dict_max(dict1: dict, dict2: dict) -> dict:
    return nested_dict_minmax(dict1, dict2, max)

@hydra.main(config_path="../../bubbleformer/config", config_name="default")    
def main(cfg: DictConfig):
    import h5py
    import json
    
    constants = NormalizerConstants(0, 0, 0, 0, 0, 0)
    fluid_params_min = None
    fluid_params_max = None
    
    # Initial loop to get the absolute max velocities
    max_velx = float("-inf")
    max_vely = float("-inf")
    for train_path in cfg.data_cfg.train_paths[:2]:
        with h5py.File(train_path, "r") as f:
            velx = f["velx"][:]
            vely = f["vely"][:]
            max_velx = max(max_velx, np.abs(velx).max().item())
            max_vely = max(max_vely, np.abs(vely).max().item())
    
    # Utility to track the variance of the velocities.
    velx_running_variance = RunningVariance(bins=1000, range=(-max_velx, max_velx))
    vely_running_variance = RunningVariance(bins=1000, range=(-max_vely, max_vely))
    
    # Loop to get the normalization constants for the SDF, temperature, and velocities.
    for train_path in cfg.data_cfg.train_paths[:2]:
        #print(train_path)
        with h5py.File(train_path, "r") as f:
            temp = f["temperature"][300:]
            velx = f["velx"][300:]
            vely = f["vely"][300:]
        with open(train_path.replace(".hdf5", ".json"), "r") as f:
            fluid_params_dict = json.load(f)
        
        x_size = fluid_params_dict["x_max"] - fluid_params_dict["x_min"]
        y_size = fluid_params_dict["y_max"] - fluid_params_dict["y_min"]
        max_domain_size = max(x_size, y_size)
        
        constants.absmax_temp = max(constants.absmax_temp, np.abs(temp).max())
        constants.max_temp_diff = max(constants.max_temp_diff, np.abs(temp).max() - fluid_params_dict["bulk_temp"])
        constants.max_domain_size = max(constants.max_domain_size, max_domain_size)
        
        velx_running_variance.update(velx)
        vely_running_variance.update(vely)
        
        if fluid_params_min is None:
            fluid_params_min = fluid_params_dict
        else:
            fluid_params_min = nested_dict_min(fluid_params_min, fluid_params_dict)
        if fluid_params_max is None:
            fluid_params_max = fluid_params_dict
        else:
            fluid_params_max = nested_dict_max(fluid_params_max, fluid_params_dict)
    
    constants.velx_std = velx_running_variance.std()
    constants.vely_std = vely_running_variance.std()
    constants.numeric_fluid_params_min = fluid_params_min
    constants.numeric_fluid_params_max = fluid_params_max

    print(constants)
    print(constants.to_yaml_string())
    
if __name__ == "__main__":
    main()