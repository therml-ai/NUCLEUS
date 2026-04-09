import torch
import numpy as np
from dataclasses import dataclass
from omegaconf import DictConfig
import hydra
import math
from typing import Optional, Callable, List
import yaml

@dataclass
class NormalizerConstants:
    max_domain_size: float
    sdf_mean: float
    sdf_std: float
    
    absmax_temp: float
    temp_mean: float
    temp_std: float
    
    velx_mean: float
    velx_std: float

    vely_mean: float
    vely_std: float

    numeric_fluid_params_min: Optional[dict] = None
    numeric_fluid_params_max: Optional[dict] = None
    
    def to_yaml_string(self) -> str:
        r"""
        This returns a YAML string that can be used as a config file for the normalizer.
        """
        fluid_params_yaml = [
            f"max_domain_size: {self.max_domain_size}",
            f"sdf_mean: {self.sdf_mean}",
            f"sdf_std: {self.sdf_std}",
            f"absmax_temp: {self.absmax_temp}",
            f"temp_mean: {self.temp_mean}",
            f"temp_std: {self.temp_std}",
            f"velx_mean: {self.velx_mean}",
            f"velx_std: {self.velx_std}",
            f"vely_mean: {self.vely_mean}",
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
    if min == max: return 0.0
    return ((value - min) / (max - min)) * 2 - 1

def minmax_unnormalize(value: float, min: float, max: float) -> float:
    if min == max: return min
    return ((value + 1) / 2) * (max - min) + min

def is_number(value: any) -> bool:
    if isinstance(value, (float, int)):
        return True
    # If it's a string, try converting to float
    try:
        float(value)
        return True
    except:
        return False

# Do not want to normalize data regarding the grid resolution.
KEYS_TO_EXCLUDE = [
    "num_blocks_x",
    "num_blocks_y",
    "nx_block",
    "ny_block",
    "dx",
    "dy",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
]

def dict_normalize_helper(dict_to_normalize: dict, func: Callable, min_dict: dict, max_dict: dict) -> dict:
    r"""
    Normalizes all numeric fields in the `dict_to_normalize`. Applied recursively to nested dictionaries.
    Non-numeric / dictionary fields are directly copied.
    """
    normalized_dict = {}
    for key, value in dict_to_normalize.items():
        if key in KEYS_TO_EXCLUDE:
            normalized_dict[key] = value
            continue
        if isinstance(value, dict):
            normalized_dict[key] = dict_normalize_helper(value, func, min_dict[key], max_dict[key])
        elif is_number(value):
            normalized_dict[key] = func(value, min_dict[key], max_dict[key])
        else:
            normalized_dict[key] = value
    return normalized_dict

class Normalizer:
    def __init__(self, constants: NormalizerConstants):
        self.constants = constants
        
    def normalize_params(self, fluid_params_dicts: List[dict]) -> List[dict]:
        return [
            dict_normalize_helper(fluid_params_dict, minmax_normalize, self.constants.numeric_fluid_params_min, self.constants.numeric_fluid_params_max)
            for fluid_params_dict in fluid_params_dicts 
        ]
    def unnormalize_params(self, fluid_params_dicts: List[dict]) -> List[dict]:
        return [
            dict_normalize_helper(fluid_params_dict, minmax_unnormalize, self.constants.numeric_fluid_params_min, self.constants.numeric_fluid_params_max)
            for fluid_params_dict in fluid_params_dicts
        ]

    def normalize_temp(self, temp: torch.Tensor, bulk_temp: torch.Tensor) -> torch.Tensor:
        pass

    def unnormalize_temp(self, temp: torch.Tensor, bulk_temp: torch.Tensor) -> torch.Tensor:
        pass

    def normalize_velx(self, vel: torch.Tensor) -> torch.Tensor:
        return (vel - self.constants.velx_mean) / self.constants.velx_std
    
    def unnormalize_velx(self, vel: torch.Tensor) -> torch.Tensor:
        return vel * self.constants.velx_std + self.constants.velx_mean
    
    def normalize_vely(self, vel: torch.Tensor) -> torch.Tensor:
        return (vel - self.constants.vely_mean) / self.constants.vely_std
    
    def unnormalize_vely(self, vel: torch.Tensor) -> torch.Tensor:
        return vel * self.constants.vely_std + self.constants.vely_mean
    
    def normalize_sdf(self, sdf: torch.Tensor) -> torch.Tensor:
        return (sdf - self.constants.sdf_mean) / self.constants.sdf_std
    
    def unnormalize_sdf(self, sdf: torch.Tensor) -> torch.Tensor:
        return sdf * self.constants.sdf_std + self.constants.sdf_mean
    
    def normalize(self, data: torch.Tensor, bulk_temp: torch.Tensor) -> torch.Tensor:
        assert data.dim() >= 4, "Data must be at least 4D (..., T, H, W, C)"
        assert data.shape[-1] == 4, "Data must have 4 channels (sdf, temp, velx, vely)"
        assert isinstance(bulk_temp, (int, float)) or data.shape[:-4] == bulk_temp.shape, "Bulk temperature must match the batch dimensions of the data"
        return torch.stack([
            self.normalize_sdf(data[..., 0]),
            self.normalize_temp(data[..., 1], bulk_temp),
            self.normalize_velx(data[..., 2]),
            self.normalize_vely(data[..., 3]),
        ], dim=-1)
        
    def unnormalize(self, data: torch.Tensor, bulk_temp: torch.Tensor) -> torch.Tensor:
        assert data.dim() >= 4, "Data must be at least 4D (..., T, H, W, C)"
        assert data.shape[-1] == 4, "Data must have 4 channels (sdf, temp, velx, vely)"
        assert isinstance(bulk_temp, (int, float)) or data.shape[:-4] == bulk_temp.shape, "Bulk temperature must match the batch dimensions of the data"
        return torch.stack([
            self.unnormalize_sdf(data[..., 0]),
            self.unnormalize_temp(data[..., 1], bulk_temp),
            self.unnormalize_velx(data[..., 2]),
            self.unnormalize_vely(data[..., 3]),
        ], dim=-1)

class StandardNormalizer(Normalizer):
    r"""
    Normalizes all fields (sdf, temperature, velocities) to have zero mean and unit variance.
    The temperature is handled difference, so that it first subtracts the samples bulk temperature,
    and then normalizes the difference to have zero mean and unit variance.
    """
    def __init__(self, constants: NormalizerConstants):
        super().__init__(constants)
        
    def normalize_temp(self, temp: torch.Tensor, bulk_temp: torch.Tensor) -> torch.Tensor:
        if not isinstance(bulk_temp, (int, float)):
            bt = bulk_temp[..., None, None, None] # (..., 1, 1, 1) for broadcasting T, H, W
        else:
            bt = bulk_temp
        return ((temp - bt) - self.constants.temp_mean) / self.constants.temp_std
    
    def unnormalize_temp(self, temp: torch.Tensor, bulk_temp: torch.Tensor) -> torch.Tensor:
        if not isinstance(bulk_temp, (int, float)):
            bt = bulk_temp[..., None, None, None] # (..., 1, 1, 1) for broadcasting T, H, W
        else:
            bt = bulk_temp
        return temp * self.constants.temp_std + self.constants.temp_mean + bt
    
class NoNormalizer(Normalizer):
    def __init__(self):
        super().__init__(None)
    
    def normalize(self, data: torch.Tensor, bulk_temp: torch.Tensor) -> torch.Tensor:
        return data
    
    def unnormalize(self, data: torch.Tensor, bulk_temp: torch.Tensor) -> torch.Tensor:
        return data
    
    def normalize_params(self, fluid_params_dicts: List[dict]) -> List[dict]:
        return fluid_params_dicts
    
    def unnormalize_params(self, fluid_params_dicts: List[dict]) -> List[dict]:
        return fluid_params_dicts
    
def get_normalizer(normalizer_cfg: dict) -> Normalizer:
    if normalizer_cfg["name"] == "standard":
        constants = NormalizerConstants(
            max_domain_size=normalizer_cfg["max_domain_size"],
            sdf_mean=normalizer_cfg["sdf_mean"],
            sdf_std=normalizer_cfg["sdf_std"],
            absmax_temp=normalizer_cfg["absmax_temp"],
            temp_mean=normalizer_cfg["temp_mean"],
            temp_std=normalizer_cfg["temp_std"],
            velx_mean=normalizer_cfg["velx_mean"],
            velx_std=normalizer_cfg["velx_std"],
            vely_mean=normalizer_cfg["vely_mean"],
            vely_std=normalizer_cfg["vely_std"],
            numeric_fluid_params_min=normalizer_cfg["fluid_params_min"],
            numeric_fluid_params_max=normalizer_cfg["fluid_params_max"],
        )
        return StandardNormalizer(constants)
    if normalizer_cfg["name"] == "no":
        return NoNormalizer()
    else:
        raise ValueError(f"Unknown normalizer: {normalizer_cfg['name']}")

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

@hydra.main(config_path="../../../config", config_name="default")    
def main(cfg: DictConfig):
    """
    This script computes and prints constants that can be used for normalizing the data.
    It prints a yaml string that can be copy-pasted into a config file and reused for training.
    """

    import h5py
    import json
    
    absmax_temp = float("-inf")
    max_domain_size = float("-inf")
    fluid_params_min = None
    fluid_params_max = None
    
    start_time = 300
    step_size = 100
    
    # Initial loop to get the limits for the running variances.
    max_sdf = float("-inf")
    max_temp = float("-inf")
    max_velx = float("-inf")
    max_vely = float("-inf")
    for train_path in cfg.data_cfg.train_paths:
        with h5py.File(train_path, "r") as f:
            sdf = f["dfun"][start_time::step_size]
            temp = f["temperature"][start_time::step_size]
            velx = f["velx"][start_time::step_size]
            vely = f["vely"][start_time::step_size]
        with open(train_path.replace(".hdf5", ".json"), "r") as f:
            fluid_params_dict = json.load(f)
        max_sdf = max(max_sdf, np.abs(sdf).max().item())
        max_temp = max(max_temp, np.abs(temp).max().item() - fluid_params_dict["bulk_temp"])
        max_velx = max(max_velx, np.abs(velx).max().item())
        max_vely = max(max_vely, np.abs(vely).max().item())
    
    sdf_running_variance = RunningVariance(bins=1000, range=(-max_sdf, max_sdf))
    temp_running_variance = RunningVariance(bins=1000, range=(-max_temp, max_temp))
    velx_running_variance = RunningVariance(bins=1000, range=(-max_velx, max_velx))
    vely_running_variance = RunningVariance(bins=1000, range=(-max_vely, max_vely))
    
    # Loop to get the normalization constants for the SDF, temperature, and velocities.
    for train_path in cfg.data_cfg.train_paths:
        #print(train_path)
        with h5py.File(train_path, "r") as f:
            sdf = f["dfun"][start_time::step_size]
            temp = f["temperature"][start_time::step_size]
            velx = f["velx"][start_time::step_size]
            vely = f["vely"][start_time::step_size]
        with open(train_path.replace(".hdf5", ".json"), "r") as f:
            fluid_params_dict = json.load(f)
        
        x_size = fluid_params_dict["x_max"] - fluid_params_dict["x_min"]
        y_size = fluid_params_dict["y_max"] - fluid_params_dict["y_min"]
        max_domain_size = max(x_size, y_size)
        
        absmax_temp = max(absmax_temp, np.abs(temp).max() - fluid_params_dict["bulk_temp"])
        max_domain_size = max(max_domain_size, max_domain_size)
        
        sdf_running_variance.update(sdf)
        temp_running_variance.update(temp - fluid_params_dict["bulk_temp"])
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
    
    constants = NormalizerConstants(
        max_domain_size=max_domain_size,
        sdf_mean=sdf_running_variance.mean(),
        sdf_std=sdf_running_variance.std(),
        absmax_temp=absmax_temp,
        temp_mean=temp_running_variance.mean(),
        temp_std=temp_running_variance.std(),
        velx_mean=velx_running_variance.mean(),
        velx_std=velx_running_variance.std(),
        vely_mean=vely_running_variance.mean(),
        vely_std=vely_running_variance.std(),
        numeric_fluid_params_min=fluid_params_min,
        numeric_fluid_params_max=fluid_params_max,
    )
    
    print(constants)
    print(constants.to_yaml_string())
    
if __name__ == "__main__":
    main()
