from typing import Tuple

import torch
import torch.nn.functional as F

from nucleus.physics.extrapolate_flux import extrapolate_phase_fluxes
from nucleus.physics.guard_cells import (
    add_halo, add_level_set_halo, add_temperature_halo, crop_halo, refill_guard_cells,
)
from nucleus.physics.phase_props import level_set_normals, min_cell_diag, smooth_density
from nucleus.physics.temp_grad import temp_gfm

# Flash-X defaults, confirmed against the stored BubbleML fields: 4 or 6 extrapolation
# iterations are ~8x worse than 5, and a smear of 1.0 reproduces rhoc 24x worse than 1.5.
DEFAULT_EXTRAP_ITERS = 5
DEFAULT_PROP_SMEAR = 1.5


def stefan_mass_flux(
    liquid_flux: torch.Tensor, vapor_flux: torch.Tensor,
    thermal_conductivity, stefan, reynolds, prandtl,
) -> torch.Tensor:
    """Stefan condition from ``Multiphase_setMassFlux`` on ``(..., H, W)`` fluxes:
    ``St / (Re * Pr) * (liquid_flux + k_gas * vapor_flux)``. Negative is evaporation."""
    return stefan / (reynolds * prandtl) * (liquid_flux + thermal_conductivity * vapor_flux)


def continuity_rhs(
    rhoc: torch.Tensor, normal_x: torch.Tensor, normal_y: torch.Tensor,
    mass_flux: torch.Tensor, dx: float, dy: float,
) -> torch.Tensor:
    """``mass_flux * n . grad(rhoc)`` with face-averaged ``rhoc``, from
    ``mph_evapDivergence2d``. All inputs and the result are ``(..., H, W)``; the
    result is zero on the 1-cell border and carries Flash-X's pressure-RHS sign,
    opposite to ``div(u)``."""
    center = (..., slice(1, -1), slice(1, -1))
    rhoc_center = rhoc[center]
    rho_right = (rhoc_center + rhoc[..., 1:-1, 2:]) / 2.0
    rho_left = (rhoc_center + rhoc[..., 1:-1, :-2]) / 2.0
    rho_up = (rhoc_center + rhoc[..., 2:, 1:-1]) / 2.0
    rho_down = (rhoc_center + rhoc[..., :-2, 1:-1]) / 2.0

    flux_x = mass_flux[center] * normal_x[center]
    flux_y = mass_flux[center] * normal_y[center]
    divergence = flux_x * (rho_right - rho_left) / dx + flux_y * (rho_up - rho_down) / dy
    return F.pad(divergence, (1, 1, 1, 1))


def _smoothed_density_and_normals(phi, dx, dy, rhogas, prop_smear, tol_normal):
    """``(rhoc, normal_x, normal_y)`` as Flash-X builds them, each ``(..., H, W)``."""
    rhoc, _ = smooth_density(phi, 1.0 / rhogas, 1.0, prop_smear * min_cell_diag(dx, dy))
    normal_x, normal_y = level_set_normals(rhoc, dx, dy, tol_normal)
    return rhoc, normal_x, normal_y


def continuity_from_mass_flux(
    mass_flux: torch.Tensor,
    sdf: torch.Tensor,
    dx: float,
    dy: float,
    rhogas,
    contact_angle=None,
    prop_smear: float = DEFAULT_PROP_SMEAR,
    tol_normal: float = 1e-13,
) -> torch.Tensor:
    """``div(u)`` from phase change, ``-mass_flux * n . grad(1/rho)``, given a known
    ``mass_flux`` (e.g. the one Flash-X stores). ``mass_flux`` and ``sdf`` are
    ``(..., H, W)``, as is the result. Same density and normals as ``continuity``."""
    # only 1-cell stencils are involved, so a 1-cell halo matches the full pipeline
    halo = 1
    phi = add_level_set_halo(sdf, halo, dy, contact_angle)
    rhoc, normal_x, normal_y = _smoothed_density_and_normals(phi, dx, dy, rhogas, prop_smear, tol_normal)
    divergence = continuity_rhs(
        refill_guard_cells(rhoc, halo), normal_x, normal_y, add_halo(mass_flux, halo), dx, dy
    )
    return -crop_halo(divergence, halo)


def _flashx_mass_transfer_pipeline(
    temp, sdf, sat_temp, dx, dy, stefan, reynolds, prandtl, thermal_conductivity, rhogas,
    wall_temp, contact_angle, n_extrap_iters, prop_smear, tol_normal, tol_temp,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flash-X's path from level set and temperature to ``Multiphase_divergence``,
    on a haloed frame cropped back. Returns ``(mass_flux, divergence)``, each
    ``(..., H, W)``."""
    # the extrapolation reaches one cell per iteration and every stencil one more
    halo = n_extrap_iters + 1
    phi = add_level_set_halo(sdf, halo, dy, contact_angle)
    temp = add_temperature_halo(temp, halo, wall_temp)
    rhoc, normal_x, normal_y = _smoothed_density_and_normals(phi, dx, dy, rhogas, prop_smear, tol_normal)

    liquid_flux, vapor_flux = temp_gfm(phi, normal_x, normal_y, temp, sat_temp, dx, dy, tol_temp)
    liquid_flux, vapor_flux = extrapolate_phase_fluxes(
        phi, normal_x, normal_y, liquid_flux, vapor_flux, dx, dy, n_extrap_iters, n_guard=halo
    )
    mass_flux = stefan_mass_flux(liquid_flux, vapor_flux, thermal_conductivity, stefan, reynolds, prandtl)
    # Flash-X evaluates rhoc on the interior and fills its guard cells by the generic
    # wall mirror, so the divergence stencil at the wall must see that mirror. rhoc
    # evaluated on the contact-angle-extended level set is right for the normals but
    # overstates grad(rhoc) at the contact line by ~25%.
    divergence = continuity_rhs(refill_guard_cells(rhoc, halo), normal_x, normal_y, mass_flux, dx, dy)
    return crop_halo(mass_flux, halo), crop_halo(divergence, halo)


def mass_transfer(
    temp: torch.Tensor,
    sdf: torch.Tensor,
    sat_temp,
    dx: float,
    dy: float,
    stefan,
    reynolds,
    prandtl,
    thermal_conductivity,
    rhogas,
    wall_temp=None,
    contact_angle=None,
    n_extrap_iters: int = DEFAULT_EXTRAP_ITERS,
    prop_smear: float = DEFAULT_PROP_SMEAR,
    tol_normal: float = 1e-13,
    tol_temp: float = 1e-2,
) -> torch.Tensor:
    """Interfacial mass flux from the Stefan condition, following Flash-X's
    ``MultiphaseEvap`` step for step. ``temp`` and ``sdf`` (``< 0`` liquid) are
    ``(..., H, W)`` with row 0 at the heater; the result has the same shape,
    negative for evaporation and nonzero only on the band the extrapolation reaches.

    ``wall_temp`` and ``contact_angle`` (degrees) enter through the heater guard
    cells; ``None`` gives a zero-gradient wall and a square contact angle. The
    material numbers are relative to the liquid; ``n_extrap_iters`` and
    ``prop_smear`` are Flash-X's ``mph_extpIt`` and ``mph_iPropSmear``."""
    mass_flux, _ = _flashx_mass_transfer_pipeline(
        temp, sdf, sat_temp, dx, dy, stefan, reynolds, prandtl, thermal_conductivity, rhogas,
        wall_temp, contact_angle, n_extrap_iters, prop_smear, tol_normal, tol_temp,
    )
    return mass_flux


def continuity(
    temp: torch.Tensor,
    sdf: torch.Tensor,
    sat_temp,
    dx: float,
    dy: float,
    stefan,
    reynolds,
    prandtl,
    thermal_conductivity,
    rhogas,
    wall_temp=None,
    contact_angle=None,
    n_extrap_iters: int = DEFAULT_EXTRAP_ITERS,
    prop_smear: float = DEFAULT_PROP_SMEAR,
    tol_normal: float = 1e-13,
    tol_temp: float = 1e-2,
) -> torch.Tensor:
    """``div(u)`` from phase change, ``-mdot * n . grad(1/rho)``, shape ``(..., H, W)``.
    Same arguments as ``mass_transfer``. Negated relative to Flash-X's pressure RHS
    so it matches the divergence of the face velocities."""
    _, divergence = _flashx_mass_transfer_pipeline(
        temp, sdf, sat_temp, dx, dy, stefan, reynolds, prandtl, thermal_conductivity, rhogas,
        wall_temp, contact_angle, n_extrap_iters, prop_smear, tol_normal, tol_temp,
    )
    return -divergence
