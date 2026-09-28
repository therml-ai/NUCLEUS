import torch
import numpy as np
from dataclasses import dataclass
from omegaconf import DictConfig
import hydra
import math
from typing import Optional, Callable, List
import yaml
from nucleus.data.layout import convert_layout

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

    # psi and phi are components of Helmholtz decomp. Only the divfree normalizer
    # sets these; standard/phase configs omit them and leave them None.
    psi_mean: Optional[float] = None
    psi_std: Optional[float] = None
    phi_mean: Optional[float] = None
    phi_std: Optional[float] = None
    
    # The divergence of the velocity field. If the model outputs
    # this directly, it benefits to normalize.
    divergence_std: Optional[float] = None

    numeric_sim_params_min: Optional[dict] = None
    numeric_sim_params_max: Optional[dict] = None

    def to_yaml_string(self) -> str:
        r"""
        This returns a YAML string that can be used as a config file for the normalizer.
        """
        sim_params_yaml = [
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

        if self.psi_mean is not None: sim_params_yaml.append(f"psi_mean: {self.psi_mean}")
        if self.psi_std is not None: sim_params_yaml.append(f"psi_std: {self.psi_std}")
        if self.phi_mean is not None: sim_params_yaml.append(f"phi_mean: {self.phi_mean}")
        if self.phi_std is not None: sim_params_yaml.append(f"phi_std: {self.phi_std}")
        if self.divergence_std is not None: sim_params_yaml.append(f"divergence_std: {self.divergence_std}")

        fmin = yaml.dump({"sim_params_min": self.numeric_sim_params_min}, default_flow_style=False) if self.numeric_sim_params_min is not None else None
        fmax = yaml.dump({"sim_params_max": self.numeric_sim_params_max}, default_flow_style=False) if self.numeric_sim_params_max is not None else None

        if fmin:
            sim_params_yaml.append(fmin)
        if fmax:
            sim_params_yaml.append(fmax)

        return "\n".join(sim_params_yaml)

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

    def normalize_scalar_with_temp(self, scalar: float, bulk_temp: float) -> float:
        pass

    def normalize_scalar_with_sdf(self, scalar: float) -> float:
        return (scalar - self.constants.sdf_mean) / self.constants.sdf_std

    def normalize_params(self, sim_params_dicts: List[dict]) -> List[dict]:
        return [
            dict_normalize_helper(sim_params_dict, minmax_normalize, self.constants.numeric_sim_params_min, self.constants.numeric_sim_params_max)
            for sim_params_dict in sim_params_dicts
        ]

    def unnormalize_params(self, sim_params_dicts: List[dict]) -> List[dict]:
        return [
            dict_normalize_helper(sim_params_dict, minmax_unnormalize, self.constants.numeric_sim_params_min, self.constants.numeric_sim_params_max)
            for sim_params_dict in sim_params_dicts
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

    def normalize_psi(self, psi: torch.Tensor) -> torch.Tensor:
        return (psi - self.constants.psi_mean) / self.constants.psi_std

    def unnormalize_psi(self, psi: torch.Tensor) -> torch.Tensor:
        return psi * self.constants.psi_std + self.constants.psi_mean

    def normalize_phi(self, phi: torch.Tensor) -> torch.Tensor:
        return (phi - self.constants.phi_mean) / self.constants.phi_std

    def unnormalize_phi(self, phi: torch.Tensor) -> torch.Tensor:
        return phi * self.constants.phi_std + self.constants.phi_mean

    def normalize(self, data: torch.Tensor, bulk_temp: torch.Tensor, layout: str = "t h w c") -> torch.Tensor:
        assert data.dim() >= 4, "Data must be at least 4D (..., T, H, W, C)"
        assert data.shape[-1] == 4, "Data must have 4 channels (sdf, temp, velx, vely)"
        assert isinstance(bulk_temp, (int, float)) or data.shape[:-4] == bulk_temp.shape, "Bulk temperature must match the batch dimensions of the data"
        # Clone once to avoid modifying the input, then normalize each channel
        # in-place to avoid the 4 intermediate tensors that torch.stack creates.
        result = data.clone()
        result[..., 0] = self.normalize_sdf(result[..., 0])
        result[..., 1] = self.normalize_temp(result[..., 1], bulk_temp)
        result[..., 2] = self.normalize_velx(result[..., 2])
        result[..., 3] = self.normalize_vely(result[..., 3])
        return result

    def unnormalize(self, data: torch.Tensor, bulk_temp: torch.Tensor, layout: str = "t h w c") -> torch.Tensor:
        assert data.dim() >= 4, "Data must be at least 4D (..., T, H, W, C)"
        if layout != "t h w c":
            data = convert_layout(data, target_layout="t h w c", source_layout=layout)
        assert data.shape[-1] == 4, "Data must have 4 channels (sdf, temp, velx, vely)"
        assert isinstance(bulk_temp, (int, float)) or data.shape[:-4] == bulk_temp.shape, "Bulk temperature must match the batch dimensions of the data"
        result = torch.stack([
            self.unnormalize_sdf(data[..., 0]),
            self.unnormalize_temp(data[..., 1], bulk_temp),
            self.unnormalize_velx(data[..., 2]),
            self.unnormalize_vely(data[..., 3]),
        ], dim=-1)
        if layout != "t h w c":
            result = convert_layout(result, target_layout=layout, source_layout="t h w c")
        return result

class StandardNormalizer(Normalizer):
    r"""
    Normalizes all fields (sdf, temperature, velocities) to have zero mean and unit variance.
    The temperature is handled difference, so that it first subtracts the samples bulk temperature,
    and then normalizes the difference to have zero mean and unit variance.
    """
    def __init__(self, constants: NormalizerConstants):
        super().__init__(constants)

    def normalize_scalar_with_temp(self, scalar: float, bulk_temp: float) -> float:
        return ((scalar - bulk_temp) - self.constants.temp_mean) / self.constants.temp_std

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

class PhaseNormalizer(StandardNormalizer):
    r"""
    This is the same as the StandardNormalizer, but does NOT normalize the SDF. This is intended
    to be used with models that process a phase mask rather than the SDF. Not normalizing the SDF
    makes it easy to convert to a phase mask, since we can just do `phase_mask = sdf > 0`. If the
    """
    def normalize_sdf(self, sdf: torch.Tensor) -> torch.Tensor:
        return sdf

    def unnormalize_sdf(self, sdf: torch.Tensor) -> torch.Tensor:
        return sdf

class DivFreeNormalizer(StandardNormalizer):
    r"""
    1. normalizes velx and vely with a shared std scalar. This allows a model
       producing divergence free outputs that can be divergence free regardless of
       whether the data is normalized or not.
    2. The velocities are already nearly zero-mean, so we just don't bother mean shifting.
    3. Normalizes the potential function psi and phi for vel = curl(psi) + grad(phi).
    """
    def __init__(self, constants: NormalizerConstants):
        super().__init__(constants)
        # Pooled std of the mean-centered components (equal-weight RMS of the two
        # per-component stds). Any single shared scalar preserves div-freeness.
        # This one keeps the combined velocity roughly unit-variance.
        self.vel_std = math.sqrt((constants.velx_std ** 2 + constants.vely_std ** 2) / 2)

    def normalize(self, data: torch.Tensor, bulk_temp: torch.Tensor, layout: str = "t h w c") -> torch.Tensor:
        assert data.dim() >= 4, "Data must be at least 4D (..., T, H, W, C)"
        assert data.shape[-1] == 6, "Data must have six channels (sdf, temp, velx, vely, psi, phi)"
        assert isinstance(bulk_temp, (int, float)) or data.shape[:-4] == bulk_temp.shape, "Bulk temperature must match the batch dimensions of the data"
        # Clone once to avoid modifying the input, then normalize each channel
        # in-place to avoid the 4 intermediate tensors that torch.stack creates.
        result = data.clone()
        result[..., 0] = self.normalize_sdf(result[..., 0])
        result[..., 1] = self.normalize_temp(result[..., 1], bulk_temp)
        result[..., 2] = self.normalize_velx(result[..., 2])
        result[..., 3] = self.normalize_vely(result[..., 3])
        result[..., 4] = self.normalize_psi(result[..., 4])
        result[..., 5] = self.normalize_phi(result[..., 5])
        return result

    def unnormalize(self, data: torch.Tensor, bulk_temp: torch.Tensor, layout: str = "t h w c") -> torch.Tensor:
        assert data.dim() >= 4, "Data must be at least 4D (..., T, H, W, C)"
        if layout != "t h w c":
            data = convert_layout(data, target_layout="t h w c", source_layout=layout)
        assert data.shape[-1] == 6, "Data must have 6 channels (sdf, temp, velx, vely, psi, phi)"
        assert isinstance(bulk_temp, (int, float)) or data.shape[:-4] == bulk_temp.shape, "Bulk temperature must match the batch dimensions of the data"
        result = torch.stack([
            self.unnormalize_sdf(data[..., 0]),
            self.unnormalize_temp(data[..., 1], bulk_temp),
            self.unnormalize_velx(data[..., 2]),
            self.unnormalize_vely(data[..., 3]),
            self.unnormalize_psi(data[..., 4]),
            self.unnormalize_phi(data[..., 5]),
        ], dim=-1)
        if layout != "t h w c":
            result = convert_layout(result, target_layout=layout, source_layout="t h w c")
        return result

    def normalize_velx(self, vel: torch.Tensor) -> torch.Tensor:
        return vel / self.vel_std

    def unnormalize_velx(self, vel: torch.Tensor) -> torch.Tensor:
        return vel * self.vel_std

    def normalize_vely(self, vel: torch.Tensor) -> torch.Tensor:
        return vel / self.vel_std

    def unnormalize_vely(self, vel: torch.Tensor) -> torch.Tensor:
        return vel * self.vel_std
    
    def normalize_divergence(self, divergence: torch.Tensor) -> torch.Tensor:
        return divergence / self.constants.divergence_std
    
    def unnormalize_divergence(self, divergence: torch.Tensor) -> torch.Tensor:
        return divergence * self.constants.divergence_std

class NoNormalizer(Normalizer):
    def __init__(self):
        super().__init__(None)

    def normalize(self, data: torch.Tensor, bulk_temp: torch.Tensor) -> torch.Tensor:
        return data

    def unnormalize(self, data: torch.Tensor, bulk_temp: torch.Tensor) -> torch.Tensor:
        return data

    def normalize_params(self, sim_params_dicts: List[dict]) -> List[dict]:
        return sim_params_dicts

    def unnormalize_params(self, sim_params_dicts: List[dict]) -> List[dict]:
        return sim_params_dicts

def get_normalizer(normalizer_cfg: dict) -> Normalizer:
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
        psi_mean=normalizer_cfg.get("psi_mean"),
        psi_std=normalizer_cfg.get("psi_std"),
        phi_mean=normalizer_cfg.get("phi_mean"),
        phi_std=normalizer_cfg.get("phi_std"),
        divergence_std=normalizer_cfg.get("divergence_std"),
        numeric_sim_params_min=normalizer_cfg["sim_params_min"],
        numeric_sim_params_max=normalizer_cfg["sim_params_max"],
    )
    if normalizer_cfg["name"] == "standard":
        return StandardNormalizer(constants)
    if normalizer_cfg["name"] == "phase":
        return PhaseNormalizer(constants)
    if normalizer_cfg["name"] == "divfree":
        return DivFreeNormalizer(constants)
    if normalizer_cfg["name"] == "no":
        return NoNormalizer()
    else:
        raise ValueError(f"Unknown normalizer: {normalizer_cfg['name']}")

class RunningVariance:
    def __init__(self):
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, value: np.ndarray):
        value = np.asarray(value, dtype=np.float64).ravel()
        batch_count = value.size
        if batch_count == 0:
            return
        batch_mean = value.mean()
        batch_m2 = ((value - batch_mean) ** 2).sum()

        # Chan's parallel merge of the running aggregate with this batch. Update the
        # mean and M2 before the count, since M2 needs the pre-merge count.
        total = self.count + batch_count
        delta = batch_mean - self._mean
        self._mean += delta * batch_count / total
        self._m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total

    def var(self) -> float:
        if self.count == 0:
            return 0.0
        return float(self._m2 / self.count)

    def std(self) -> float:
        return math.sqrt(self.var())

    def mean(self) -> float:
        return float(self._mean)

def nested_dict_minmax(dict1: dict, dict2: dict, op: Callable) -> dict:
    r"""
    Applies a reduction operation `op` to two nested dictionaries (i.e., potentially a dict of dicts). This assumes that the
    dictionaries have identical structure.
    NOTE: This only applies to numeric values, so strings are excluded from the output.
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

@hydra.main(config_path="../../../config", config_name="default", version_base="1.1")
def main(cfg: DictConfig):
    """
    This script computes and prints constants that can be used for normalizing the data.
    It prints a yaml string that can be copy-pasted into a config file and reused for training.
    """

    import h5py
    import json

    from nucleus.physics.poisson import helmholtz_from_faces
    from nucleus.physics.poisson import divergence_centers_from_faces

    absmax_temp = float("-inf")
    max_domain_size = float("-inf")
    sim_params_min = None
    sim_params_max = None

    start_time = 300
    step_size = 100

    sdf_running_variance = RunningVariance()
    temp_running_variance = RunningVariance()
    velx_running_variance = RunningVariance()
    vely_running_variance = RunningVariance()
    divergence_running_variance = RunningVariance()
    helmholtz_psi_variance = RunningVariance()
    helmholtz_phi_variance = RunningVariance()

    for train_path in cfg.data_cfg.train_paths:
        with h5py.File(train_path, "r") as f:
            velface = "velfacex" in f.keys()
            VELX = "velfacex" if velface else "velx"
            VELY = "velfacey" if velface else "vely"
            sdf = f["dfun"][start_time::step_size]
            temp = f["temperature"][start_time::step_size]
            velx = f[VELX][start_time::step_size]
            vely = f[VELY][start_time::step_size]
        with open(train_path.replace(".hdf5", ".json"), "r") as f:
            sim_params_dict = json.load(f)

        x_size = sim_params_dict["x_max"] - sim_params_dict["x_min"]
        y_size = sim_params_dict["y_max"] - sim_params_dict["y_min"]
        max_domain_size = max(max_domain_size, x_size, y_size)

        absmax_temp = max(absmax_temp, np.abs(temp).max() - sim_params_dict["bulk_temp"])

        sdf_running_variance.update(sdf)
        temp_running_variance.update(temp - sim_params_dict["bulk_temp"])
        velx_running_variance.update(velx)
        vely_running_variance.update(vely)

        if velface:
            psi_nodes, phi_centers = helmholtz_from_faces(velx, vely, 1/32, 1/32)
            helmholtz_psi_variance.update(psi_nodes)
            helmholtz_phi_variance.update(phi_centers)
            div = divergence_centers_from_faces(velx, vely, 1/32, 1/32)
            divergence_running_variance.update(div)

        if sim_params_min is None:
            sim_params_min = sim_params_dict
        else:
            sim_params_min = nested_dict_min(sim_params_min, sim_params_dict)
        if sim_params_max is None:
            sim_params_max = sim_params_dict
        else:
            sim_params_max = nested_dict_max(sim_params_max, sim_params_dict)

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
        psi_mean=helmholtz_psi_variance.mean() if helmholtz_psi_variance.count > 0 else None,
        psi_std=helmholtz_psi_variance.std() if helmholtz_psi_variance.count > 0 else None,
        phi_mean=helmholtz_phi_variance.mean() if helmholtz_phi_variance.count > 0 else None,
        phi_std=helmholtz_phi_variance.std() if helmholtz_phi_variance.count > 0 else None,
        divergence_std=divergence_running_variance.std() if divergence_running_variance.count > 0 else None,
        numeric_sim_params_min=sim_params_min,
        numeric_sim_params_max=sim_params_max,
    )

    print(constants)
    print(constants.to_yaml_string())

if __name__ == "__main__":
    main()
