import torch

from nucleus.physics.poisson import GRID_SPACING


def interface_decay(sdf: torch.Tensor, band_width: float = 4.0 * GRID_SPACING) -> torch.Tensor:
    ramp = (sdf.abs() / band_width).clamp(max=1.0)
    return 1.0 - (1.0 - ramp) ** 2


def heater_decay(
    x_coords: torch.Tensor,
    height: int,
    dy: float,
    heater_x_min: float,
    heater_x_max: float,
    band_width: float,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    x_coords = x_coords.to(device=device, dtype=dtype)
    y_distance = ((torch.arange(height, device=device, dtype=dtype) + 0.5) * dy).reshape(height, 1)
    # horizontal distance past the heater's x-extent (0 while over the heater)
    x_outside = (heater_x_min - x_coords).clamp(min=0.0) + (x_coords - heater_x_max).clamp(min=0.0)
    distance = torch.sqrt(x_outside**2 + y_distance**2)
    return interface_decay(distance, band_width)


def temperature_ansatz(
    nn: torch.Tensor,
    sdf: torch.Tensor,
    saturation_temperature,
    band_width: float = 2.0 * GRID_SPACING,
    heater_temperature=None,
    x_coords: torch.Tensor = None,
    heater_x_min: float = None,
    heater_x_max: float = None,
    heater_band_width: float = 2.0 * GRID_SPACING,
    dy: float = GRID_SPACING,
) -> torch.Tensor:
    field = saturation_temperature + interface_decay(sdf, band_width) * nn
    if heater_temperature is None:
        return field
    assert x_coords is not None and heater_x_min is not None and heater_x_max is not None, (
        "heater_temperature requires x_coords, heater_x_min, heater_x_max"
    )
    heater = heater_decay(
        x_coords, sdf.shape[-2], dy, heater_x_min, heater_x_max,
        heater_band_width, sdf.device, sdf.dtype,
    )
    return heater_temperature + heater * (field - heater_temperature)
