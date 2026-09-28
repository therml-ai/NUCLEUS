from typing import Tuple

import torch
import torch.nn.functional as F

from nucleus.physics.guard_cells import refill_guard_cells
from nucleus.physics.phase_props import sign_heaviside


def advect_upwind_rhs(
    field: torch.Tensor, vel_x: torch.Tensor, vel_y: torch.Tensor, dx: float, dy: float
) -> torch.Tensor:
    """``-(vel_x, vel_y) . grad(field)`` with first-order upwinding, from
    ``Stencils_cnt_advectUpwind2d``. All ``(..., H, W)``; zero on the 1-cell border."""
    center = (..., slice(1, -1), slice(1, -1))
    field_center = field[center]
    vel_x_center, vel_y_center = vel_x[center], vel_y[center]

    forward_x = field[..., 1:-1, 2:] - field_center
    backward_x = field_center - field[..., 1:-1, :-2]
    forward_y = field[..., 2:, 1:-1] - field_center
    backward_y = field_center - field[..., :-2, 1:-1]

    rhs = (
        -(vel_x_center.clamp_min(0.0) * backward_x + vel_x_center.clamp_max(0.0) * forward_x) / dx
        - (vel_y_center.clamp_min(0.0) * backward_y + vel_y_center.clamp_max(0.0) * forward_y) / dy
    )
    return F.pad(rhs, (1, 1, 1, 1))


def phased_update(
    field: torch.Tensor, rhs: torch.Tensor, phi: torch.Tensor, dt: float
) -> torch.Tensor:
    """Euler step of ``field`` by ``rhs`` where ``phi >= 0``, from ``mph_phasedFluxes``.
    All ``(..., H, W)``."""
    return field + dt * rhs * sign_heaviside(phi)


def extrapolate_phase_fluxes(
    phi: torch.Tensor, normal_x: torch.Tensor, normal_y: torch.Tensor,
    liquid_flux: torch.Tensor, vapor_flux: torch.Tensor, dx: float, dy: float,
    n_iterations: int = 5, n_guard: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Push the one-sided fluxes across the interface, from the
    ``Multiphase_extrapFluxes`` loop: ``liquid_flux`` along ``+normal`` into the
    vapor, ``vapor_flux`` along ``-normal`` into the liquid, ``n_iterations`` upwind
    steps each. All fields ``(..., H, W)``; returns ``(liquid_flux, vapor_flux)``.

    This is a partial extension, not a steady state: ``n_iterations`` (Flash-X
    ``mph_extpIt``) sets the band width. ``n_guard`` is the caller's halo width."""
    # the Fortran uses del(IAXIS) for both directions
    dt = 0.5 * dx
    for _ in range(n_iterations):
        # Flash-X refills guard cells every iteration, so the halo must be re-derived
        # from the interior rather than advected on its own
        liquid_flux = refill_guard_cells(liquid_flux, n_guard)
        vapor_flux = refill_guard_cells(vapor_flux, n_guard)

        liquid_rhs = advect_upwind_rhs(liquid_flux, normal_x, normal_y, dx, dy)
        liquid_flux = phased_update(liquid_flux, liquid_rhs, phi, dt)

        vapor_rhs = advect_upwind_rhs(vapor_flux, -normal_x, -normal_y, dx, dy)
        vapor_flux = phased_update(vapor_flux, vapor_rhs, -phi, dt)
    return refill_guard_cells(liquid_flux, n_guard), refill_guard_cells(vapor_flux, n_guard)
