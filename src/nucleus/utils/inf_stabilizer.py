import torch

def clip_temp_by_phase(temp, sdf, sat_temp: float, wall_temp: float):
    """Clamp temperature by phase using physical (unnormalized) fields.

    The interface is the SDF zero level set, so vapor is ``sdf > 0``. Temperature
    never exceeds the heater wall temperature; vapor is floored at the saturation
    temperature, while liquid near the heater may sit above it.
    """
    temp = torch.clamp(temp, max=wall_temp)
    temp = torch.where(sdf > 0, torch.clamp(temp, min=sat_temp), temp)
    return temp
