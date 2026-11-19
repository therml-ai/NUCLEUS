from typing import Tuple

import torch


def _central_difference(field: torch.Tensor, spacing: float, dim: int) -> torch.Tensor:
    """Central difference along ``dim``, one-sided at the two boundary cells."""
    # narrow + cat rather than in-place writes so it differentiates and compiles cleanly
    n = field.size(dim)
    interior = (field.narrow(dim, 2, n - 2) - field.narrow(dim, 0, n - 2)) / (2.0 * spacing)
    first = (field.narrow(dim, 1, 1) - field.narrow(dim, 0, 1)) / spacing
    last = (field.narrow(dim, n - 1, 1) - field.narrow(dim, n - 2, 1)) / spacing
    return torch.cat([first, interior, last], dim=dim)


def interface_normals(
    sdf: torch.Tensor, dx: float, dy: float, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``grad(sdf) / |grad(sdf)|`` on ``(..., H, W)``; returns ``(normal_x, normal_y)``
    of the same shape. ``eps`` floors the magnitude so flat regions give a near-zero
    vector instead of dividing by zero."""
    grad_x = _central_difference(sdf, dx, dim=-1)
    grad_y = _central_difference(sdf, dy, dim=-2)
    magnitude = torch.sqrt(grad_x**2 + grad_y**2).clamp_min(eps)
    return grad_x / magnitude, grad_y / magnitude


def interface_mask(sdf):
    """Cells with a 4-neighbour in the other phase. ``(..., H, W)`` -> bool of the
    same shape."""
    assert sdf.dim() >= 2, "SDF must be of shape (..., H, W)"
    signs = torch.sign(sdf)
    interface = torch.zeros_like(sdf, dtype=torch.bool)

    # Inequality is symmetric, so one comparison per axis marks both neighbors.
    rows_differ = signs[..., :-1, :] != signs[..., 1:, :]
    interface[..., :-1, :] |= rows_differ
    interface[..., 1:, :] |= rows_differ

    cols_differ = signs[..., :, :-1] != signs[..., :, 1:]
    interface[..., :, :-1] |= cols_differ
    interface[..., :, 1:] |= cols_differ
    return interface


def vapor_mask(sdf):
    return sdf >= 0


def liquid_mask(sdf):
    return sdf < 0


def band_mask(sdf: torch.Tensor, band_width: float) -> torch.Tensor:
    """``|sdf| <= band_width`` on ``(..., H, W)``."""
    return sdf.abs() <= band_width

