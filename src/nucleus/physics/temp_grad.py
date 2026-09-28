from typing import Tuple

import torch
import torch.nn.functional as F

from nucleus.physics.phase_props import sign_heaviside

# Ghost-fluid one-sided temperature gradients, ported from mph_tempGfm2d.F90. The
# temperature is continuous but kinked at the interface, so a neighbour across it is
# replaced by this side's linear extrapolation through T_sat:
#
#     ghost(T)_{i+1} = T_i + (T_sat - T_i) / theta,   theta = |phi_i| / (|phi_i| + |phi_{i+1}|)


def _theta(phi_center: torch.Tensor, phi_neighbor: torch.Tensor, tol: float) -> torch.Tensor:
    """``max(tol, |phi_c| / (|phi_c| + |phi_nb|))``."""
    # Both cells exactly on the interface gives 0/0; gfortran's max(tol, NaN) yields
    # tol, and the NaN must not reach torch.where or it poisons the backward pass.
    denominator = phi_center.abs() + phi_neighbor.abs()
    is_finite = denominator > 0
    safe_denominator = torch.where(is_finite, denominator, torch.ones_like(denominator))
    ratio = torch.where(is_finite, phi_center.abs() / safe_denominator, torch.full_like(denominator, tol))
    return ratio.clamp_min(tol)


def _ghost_neighbor(
    phi_center: torch.Tensor, phi_neighbor: torch.Tensor,
    temp_center: torch.Tensor, temp_neighbor: torch.Tensor, sat_temp, tol: float,
) -> torch.Tensor:
    """The real neighbour temperature, or the ghost through ``sat_temp`` when the
    neighbour is across the interface."""
    across = phi_center * phi_neighbor <= 0.0
    ghost = temp_center + (sat_temp - temp_center) / _theta(phi_center, phi_neighbor, tol)
    return torch.where(across, ghost, temp_neighbor)


def temp_gfm(
    phi: torch.Tensor, normal_x: torch.Tensor, normal_y: torch.Tensor,
    temp: torch.Tensor, sat_temp, dx: float, dy: float, tol: float = 1e-2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-sided ``n . grad(T)`` at the interface (Flash-X ``HFLQ_VAR``/``HFGS_VAR``).
    All fields ``(..., H, W)``. Returns ``(liquid_flux, vapor_flux)``: ``n . grad(T)``
    on liquid cells and ``-n . grad(T)`` on vapor cells, each zero on the other phase
    and on the 1-cell border. ``tol`` floors the sub-cell interface distance."""
    center = (..., slice(1, -1), slice(1, -1))
    phi_center, temp_center = phi[center], temp[center]

    right = _ghost_neighbor(phi_center, phi[..., 1:-1, 2:], temp_center, temp[..., 1:-1, 2:], sat_temp, tol)
    left = _ghost_neighbor(phi_center, phi[..., 1:-1, :-2], temp_center, temp[..., 1:-1, :-2], sat_temp, tol)
    up = _ghost_neighbor(phi_center, phi[..., 2:, 1:-1], temp_center, temp[..., 2:, 1:-1], sat_temp, tol)
    down = _ghost_neighbor(phi_center, phi[..., :-2, 1:-1], temp_center, temp[..., :-2, 1:-1], sat_temp, tol)

    grad_x = (right - left) / (2.0 * dx)
    grad_y = (up - down) / (2.0 * dy)
    normal_gradient = normal_x[center] * grad_x + normal_y[center] * grad_y

    vapor = sign_heaviside(phi_center)
    liquid_flux = (1.0 - vapor) * normal_gradient
    vapor_flux = vapor * (-normal_gradient)
    return F.pad(liquid_flux, (1, 1, 1, 1)), F.pad(vapor_flux, (1, 1, 1, 1))
