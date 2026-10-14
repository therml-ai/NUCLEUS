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


def _heater_rows(padded: torch.Tensor, halo: int, heater_rows: torch.Tensor, heater_mask=None) -> torch.Tensor:
    """``padded`` with its bottom ``halo`` rows replaced by ``heater_rows`` over the
    heater's columns. ``heater_mask`` ``(..., 1, W)`` marks them; ``None`` is the whole
    wall. As in Flash-X's ``Heater_applyBCToRegion``, a guard cell is on the heater
    when its x lies strictly within the heater, so the halo's columns past the domain
    edges never are."""
    if heater_mask is not None:
        outside = torch.zeros(*heater_mask.shape[:-1], halo, dtype=torch.bool, device=heater_mask.device)
        on_heater = torch.cat([outside, heater_mask.to(torch.bool), outside], dim=-1)
        heater_rows = torch.where(on_heater, heater_rows, padded[..., :halo, :])
    return torch.cat([heater_rows, padded[..., halo:, :]], dim=-2)


def add_temperature_halo(temp: torch.Tensor, halo: int, wall_temp=None, heater_mask=None) -> torch.Tensor:
    """``add_halo`` for ``temp`` ``(..., H, W)``, with the heater rows reflected about
    ``wall_temp`` (scalar or ``(..., halo, W)``) so stencils see a Dirichlet wall.
    ``None`` keeps the zero-gradient wall replication gives, which is also the wall
    off the heater when ``heater_mask`` (see ``_heater_rows``) is given."""
    padded = add_halo(temp, halo)
    if halo == 0 or wall_temp is None:
        return padded
    reflected = 2.0 * wall_temp - padded[..., halo:2 * halo, :].flip(-2)
    return _heater_rows(padded, halo, reflected, heater_mask)


def add_level_set_halo(
    sdf: torch.Tensor, halo: int, dy: float, contact_angle_degrees=None, heater_mask=None
) -> torch.Tensor:
    """``add_halo`` for ``sdf`` ``(..., H, W)``, with the heater rows extended at the
    contact-angle slope ``d(phi)/dn = cos(theta)``. ``None`` replicates, which is the
    ``theta = 90`` case, and also the wall off the heater when ``heater_mask`` (see
    ``_heater_rows``) is given."""
    padded = add_halo(sdf, halo)
    if halo == 0 or contact_angle_degrees is None:
        return padded
    angle = torch.as_tensor(contact_angle_degrees, dtype=sdf.dtype, device=sdf.device)
    step = dy * torch.cos(torch.deg2rad(angle))
    rows = torch.arange(halo, 0, -1, dtype=sdf.dtype, device=sdf.device).reshape(-1, 1)
    ghost = padded[..., halo:halo + 1, :] - rows * step
    return _heater_rows(padded, halo, ghost, heater_mask)


def refill_guard_cells(field: torch.Tensor, halo: int) -> torch.Tensor:
    """Stand-in for ``Grid_fillGuardCells``: rewrite the outer ``halo`` ring of
    ``field`` ``(..., H, W)`` from its interior by replication."""
    return field if halo == 0 else _pad_replicate(crop_halo(field, halo), halo)
