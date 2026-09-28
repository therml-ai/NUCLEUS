from typing import Tuple

import torch


def _central_difference(field: torch.Tensor, spacing: float, dim: int) -> torch.Tensor:
    """Explicit finite-difference derivative of a cell-centered field along ``dim``:
    a second-order central difference ``(f[i+1] - f[i-1]) / (2h)`` in the interior and
    a first-order one-sided difference at the two boundary cells. Built with
    ``narrow`` + ``cat`` (no in-place writes) so it differentiates and compiles cleanly.
    """
    n = field.size(dim)
    interior = (field.narrow(dim, 2, n - 2) - field.narrow(dim, 0, n - 2)) / (2.0 * spacing)
    first = (field.narrow(dim, 1, 1) - field.narrow(dim, 0, 1)) / spacing
    last = (field.narrow(dim, n - 1, 1) - field.narrow(dim, n - 2, 1)) / spacing
    return torch.cat([first, interior, last], dim=dim)


def interface_normals(
    sdf: torch.Tensor, dx: float, dy: float, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """normal direction of the SDF, which can be used to get the normal of the
       interface (the zero level set of the SDF).
    Args:
        sdf: cell-centered signed distance, shape ``(..., H, W)``. x is the last
            axis (width) and y the second-to-last (height).
        dx: cell spacing in x.
        dy: cell spacing in y.
        eps: floor on ``|grad(sdf)|``. Where the gradient vanishes -- flat regions
            far from the interface, or local extrema -- the normal is undefined;
            the floor makes those cells return a near-zero vector rather than
            dividing by zero.

    Returns:
        ``(normal_x, normal_y)``, each shape ``(..., H, W)``.
    """
    grad_x = _central_difference(sdf, dx, dim=-1)
    grad_y = _central_difference(sdf, dy, dim=-2)
    magnitude = torch.sqrt(grad_x**2 + grad_y**2).clamp_min(eps)
    return grad_x / magnitude, grad_y / magnitude


def interface_mask(sdf):
    r"""
    Cells adjacent to the zero level set: a cell is marked when any of its four
    neighbors lies in the other phase (the sign of the SDF differs). Works for any
    shape (..., H, W) and stays on the input's device.
    """
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
    """Cells whose center lies within ``band_width`` (a distance, in the SDF's
    units) of the interface, i.e. ``|sdf| <= band_width``. Shape ``(..., H, W)``.
    """
    return sdf.abs() <= band_width

