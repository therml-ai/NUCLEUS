import torch
import torch.nn.functional as F

# The Flash-X stencils ported here read neighbours out of guard cells; the ports
# leave a zero ring instead, so frames are grown by a halo before computing and
# cropped after. Row 0 of a (H, W) frame is the heater wall.


def _pad_replicate(field: torch.Tensor, halo: int) -> torch.Tensor:
    """Replicate-pad ``(..., H, W)`` -> ``(..., H + 2*halo, W + 2*halo)``."""
    # F.pad replicate only accepts 3-5D input, hence the reshape round trip
    padded = F.pad(field.reshape(-1, 1, *field.shape[-2:]), (halo,) * 4, mode="replicate")
    return padded.reshape(*field.shape[:-2], *padded.shape[-2:])


def add_halo(field: torch.Tensor, halo: int) -> torch.Tensor:
    """``(..., H, W)`` -> ``(..., H + 2*halo, W + 2*halo)`` by edge replication."""
    return field if halo == 0 else _pad_replicate(field, halo)


def crop_halo(field: torch.Tensor, halo: int) -> torch.Tensor:
    """Inverse of ``add_halo``: ``(..., H + 2*halo, W + 2*halo)`` -> ``(..., H, W)``."""
    return field if halo == 0 else field[..., halo:-halo, halo:-halo]


def add_temperature_halo(temp: torch.Tensor, halo: int, wall_temp=None) -> torch.Tensor:
    """``add_halo`` for ``temp`` ``(..., H, W)``, with the heater rows reflected about
    ``wall_temp`` (scalar or ``(..., halo, W)``) so stencils see a Dirichlet wall.
    ``None`` keeps the zero-gradient wall replication gives."""
    padded = add_halo(temp, halo)
    if halo == 0 or wall_temp is None:
        return padded
    reflected = 2.0 * wall_temp - padded[..., halo:2 * halo, :].flip(-2)
    return torch.cat([reflected, padded[..., halo:, :]], dim=-2)


def add_level_set_halo(
    sdf: torch.Tensor, halo: int, dy: float, contact_angle_degrees=None
) -> torch.Tensor:
    """``add_halo`` for ``sdf`` ``(..., H, W)``, with the heater rows extended at the
    contact-angle slope ``d(phi)/dn = cos(theta)``. ``None`` replicates, which is the
    ``theta = 90`` case."""
    padded = add_halo(sdf, halo)
    if halo == 0 or contact_angle_degrees is None:
        return padded
    angle = torch.as_tensor(contact_angle_degrees, dtype=sdf.dtype, device=sdf.device)
    step = dy * torch.cos(torch.deg2rad(angle))
    rows = torch.arange(halo, 0, -1, dtype=sdf.dtype, device=sdf.device).reshape(-1, 1)
    ghost = padded[..., halo:halo + 1, :] - rows * step
    return torch.cat([ghost, padded[..., halo:, :]], dim=-2)


def refill_guard_cells(field: torch.Tensor, halo: int) -> torch.Tensor:
    """Stand-in for ``Grid_fillGuardCells``: rewrite the outer ``halo`` ring of
    ``field`` ``(..., H, W)`` from its interior by replication."""
    return field if halo == 0 else _pad_replicate(crop_halo(field, halo), halo)
