from typing import Optional

import torch

from nucleus.physics.guard_cells import add_level_set_halo

# inside the gradient magnitudes' sqrt, whose gradient is infinite at zero; zero
# upwind differences are common, and inf * 0 makes the backward pass NaN
SQRT_EPS = 1e-12

def sdf_reinit_drift(
    sdf_before: torch.Tensor,
    sdf_after:  torch.Tensor,
    dx: float,
) -> float:
    """
    Measures how much the zero level set moved after redistancing. Should be much less than dx.
    """
    near_interface = torch.abs(sdf_before) < 3 * dx
    if not near_interface.any():
        return 0.0
    drift = torch.abs(sdf_after[near_interface] - sdf_before[near_interface])
    return drift.mean().item()

def verify_sdf(sdf, dx, dy=None):
    if dy is None: dy = dx
    # Verify it's a distance function. |grad(sdf)| mean should be ~1, std should be ~0.
    grad_y, grad_x = torch.gradient(sdf, spacing=(dy, dx), dim=(-2, -1), edge_order=1)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    return grad_magnitude.mean(dim=(-2, -1)), grad_magnitude.std(dim=(-2, -1))

def sdf_reinit_sussman(
    sdf0: torch.Tensor,
    dx: float,
    dy: float = None,
    n_iter: int = 5,
    dtau: float = None,
    contact_angle=None,
    heater_mask=None,
    reinit_band: Optional[float] = None,
) -> torch.Tensor:
    """
    Redistance ``sdf0`` ``(..., H, W)`` toward ``|grad| = 1`` by Sussman's pseudo-time
    iteration, with Russo & Smereka's (2000) subcell fix holding the zero level set
    where ``sdf0`` puts it. Convention: sdf > 0 = vapor, sdf < 0 = liquid.
    Args:
        dx, dy: grid spacing in x (along W) and y (along H); dy defaults to dx
        n_iter: pseudo-time iterations; each reaches about half a cell further out
        dtau:   pseudo-timestep (defaults to 0.5 * min(dx, dy))
        contact_angle: degrees, scalar or broadcastable to ``(..., 1, 1)``; the heater
            wall (row 0) holds ``d(sdf)/dn = cos(contact_angle)``, as in
            ``add_level_set_halo``. ``None`` is a 90 degree wall.
        heater_mask: ``(..., 1, W)``, the heater's columns; ``None`` is the whole wall
        reinit_band: only the cells with ``|sdf0| < reinit_band`` are redistanced, the
            rest keep sdf0; ``None`` redistances every cell
    """
    dy = dx if dy is None else dy
    dtau = 0.5 * min(dx, dy) if dtau is None else dtau
    # The stencils read the wall's ghost cells, so the contact line keeps the angle
    # Flash-X imposes rather than the zero slope an edge copy gives.
    with_halo = lambda field: add_level_set_halo(field, 1, dy, contact_angle, heater_mask)
    sign0 = torch.sign(sdf0)
    padded0 = with_halo(sdf0)
    interface = _interface_cells(padded0)
    distance = _distance_to_interface(padded0, dx, dy)
    in_band = torch.ones_like(interface) if reinit_band is None else sdf0.abs() < reinit_band
    sdf = sdf0
    for _ in range(n_iter):
        # Upwind differences across the interface would move it, so the cells on it
        # relax toward the distance sdf0 gives them instead.
        subcell = (sign0 * sdf.abs() - distance) / min(dx, dy)
        upwind = sign0 * (_godunov_grad_mag(with_halo(sdf), sign0, dx, dy) - 1.0)
        update = torch.where(interface, subcell, upwind)
        sdf = sdf - dtau * torch.where(in_band, update, torch.zeros_like(update))
    return sdf


def _neighbor(padded: torch.Tensor, dim: int, step: int) -> torch.Tensor:
    """From ``padded`` ``(..., H + 2, W + 2)``, the ``(..., H, W)`` values at
    index + step along dim (-1 or -2)."""
    row_step, col_step = (0, step) if dim == -1 else (step, 0)
    height, width = padded.shape[-2] - 2, padded.shape[-1] - 2
    return padded[..., 1 + row_step:1 + row_step + height, 1 + col_step:1 + col_step + width]


def _interface_cells(padded: torch.Tensor) -> torch.Tensor:
    """The cells whose sign differs from, or touches zero with, one of their four neighbors."""
    center = _neighbor(padded, -1, 0)
    crossings = [center * _neighbor(padded, dim, step) <= 0 for dim in (-1, -2) for step in (-1, 1)]
    return torch.stack(crossings).any(dim=0)


def _distance_to_interface(padded: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """sdf / |grad sdf|, with the largest of the central, forward and backward gradient
    estimates, as Russo & Smereka estimate a cell's signed distance to the interface."""
    estimates = []
    for low, high, scale in ((-1, 1, 2.0), (0, 1, 1.0), (-1, 0, 1.0)):
        grad_x = (_neighbor(padded, -1, high) - _neighbor(padded, -1, low)) / (scale * dx)
        grad_y = (_neighbor(padded, -2, high) - _neighbor(padded, -2, low)) / (scale * dy)
        estimates.append(torch.sqrt(grad_x**2 + grad_y**2 + SQRT_EPS))
    grad_mag = torch.stack(estimates).amax(dim=0)
    return _neighbor(padded, -1, 0) / grad_mag


def _godunov_grad_mag(padded: torch.Tensor, sign: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """|grad sdf| from the upwind one-sided differences: per axis, the larger of the
    two that carry information outward from the interface."""
    center = _neighbor(padded, -1, 0)
    terms = []
    for dim, spacing in ((-1, dx), (-2, dy)):
        backward = (center - _neighbor(padded, dim, -1)) / spacing
        forward = (_neighbor(padded, dim, 1) - center) / spacing
        outward = torch.maximum(torch.relu(backward) ** 2, torch.relu(-forward) ** 2)
        inward = torch.maximum(torch.relu(-backward) ** 2, torch.relu(forward) ** 2)
        terms.append(torch.where(sign > 0, outward, inward))
    return torch.sqrt(terms[0] + terms[1] + SQRT_EPS)
