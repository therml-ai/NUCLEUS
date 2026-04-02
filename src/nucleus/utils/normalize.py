import torch

def normalize_temp(temp: torch.Tensor, bulk_temp: torch.Tensor, heater_temp: torch.Tensor) -> torch.Tensor:
    assert (heater_temp > bulk_temp).all(), "Heater temperature must be greater than bulk temperature"
    
    normalized_temp = (temp - bulk_temp) / (heater_temp - bulk_temp)
    
    normalized_temp * (heater_temp - bulk_temp) + bulk_temp
    
    
    
    bulk_temp = bulk_temp[:, None, None, None]
    heater_temp = heater_temp[:, None, None, None]
    temp = torch.clip(temp, min=bulk_temp, max=heater_temp)
    # map temperature range to [0, heater_temp - bulk_temp] based on heater and bulk temperature
    return temp - bulk_temp

def unnormalize_temp(temp: torch.Tensor, bulk_temp: torch.Tensor, heater_temp: torch.Tensor) -> torch.Tensor:
    bulk_temp = bulk_temp[:, None, None, None]
    return temp + bulk_temp

def normalize_sdf(sdf: torch.Tensor) -> torch.Tensor:
    # the pool boiling domain is 16x16, so this is approximately normalizing to [-1, 1]
    return sdf / 16

def unnormalize_sdf(sdf: torch.Tensor) -> torch.Tensor:
    return sdf * 16

def normalize_vel(vel: torch.Tensor) -> torch.Tensor:
    # velocities are already pretty small and approximately Gaussian, so not normalizing.
    return vel

def unnormalize_vel(vel: torch.Tensor) -> torch.Tensor:
    return vel

def normalize(x: torch.Tensor, bulk_temp, heater_temp) -> torch.Tensor:
    assert x.dim() == 5, "Input must be a 5D tensor (B, T, C, H, W)"
    return torch.stack([
        normalize_sdf(x[:, :, 0, :, :]),
        normalize_temp(x[:, :, 1, :, :], bulk_temp, heater_temp),
        normalize_vel(x[:, :, 2, :, :]),
        normalize_vel(x[:, :, 3, :, :]),
    ], dim=2)

def unnormalize(x: torch.Tensor, bulk_temp, heater_temp) -> torch.Tensor:
    assert x.dim() == 5, "Input must be a 5D tensor (B, T, C, H, W)"
    return torch.stack([
        unnormalize_sdf(x[:, :, 0, :, :]),
        unnormalize_temp(x[:, :, 1, :, :], bulk_temp, heater_temp),
        unnormalize_vel(x[:, :, 2, :, :]),
        unnormalize_vel(x[:, :, 3, :, :]),
    ], dim=2)