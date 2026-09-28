from typing import Tuple
import math

import torch

from nucleus.physics.sdf import (
    band_mask,
    interface_normals,
    interface_mask,
    liquid_mask,
    vapor_mask,
)
from nucleus.physics.temp_grad import vapor_temp_grad, liquid_temp_grad
from nucleus.physics.extrapolate_flux import extrapolate_phase_flux

DEFAULT_BAND_CELLS = 4


def interface_heatflux(
    temp: torch.Tensor, sdf: torch.Tensor, sat_temp, dx: float, dy: float,
    band_cells: int = DEFAULT_BAND_CELLS, wall_temp=None, eps: float = 1e-13,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normal temperature gradient ``grad(T) . n`` on each side of the liquid/vapor interface.

    Args:
        temp: cell-centered temperature, shape ``(..., H, W)``.
        sdf: cell-centered signed distance, shape ``(..., H, W)``. sdf < 0 is liquid,
            sdf >= 0 is vapor.
        sat_temp: interface (saturation) temperature; scalar or broadcastable to
            ``temp``. Passed to the ghost-fluid gradients.
        dx: cell spacing in x.
        dy: cell spacing in y.
        band_cells: how many cells each one-sided gradient is extrapolated into the
            opposite phase, setting the width of the overlap band.
        wall_temp: heater temperature at the bottom wall, in the same units as
            ``temp``; applies a Dirichlet BC to the ghost-fluid gradients there.
            ``None`` keeps a zero-gradient wall.
        eps: floor passed through to ``interface_normals`` and the ghost fluid
            gradients.

    Returns:
        ``(liquid_side, vapor_side)``, each shape ``(..., H, W)``, holding
        ``grad(T) . n`` for that phase on the ``band_cells``-wide band and zero
        outside it, so the two overlap on the band and can be subtracted there.
    """
    normal_x, normal_y = interface_normals(sdf, dx, dy, eps)
    
    liquid_grad_x, liquid_grad_y = liquid_temp_grad(temp, sdf, sat_temp, dx, dy, wall_temp, eps)
    liquid_heat_flux = liquid_grad_x * normal_x + liquid_grad_y * normal_y

    vapor_grad_x, vapor_grad_y = vapor_temp_grad(temp, sdf, sat_temp, dx, dy, wall_temp, eps)
    vapor_heat_flux = vapor_grad_x * normal_x + vapor_grad_y * normal_y
        
    ext_liquid_heat_flux, ext_vapor_heat_flux = extrapolate_phase_flux(
        liquid_heat_flux, vapor_heat_flux, sdf, normal_x, normal_y, dx, dy)

    lmask = liquid_mask(sdf).to(temp.dtype)
    vmask = vapor_mask(sdf).to(temp.dtype)
    
    #band_mask = (abs(sdf) < (band_cells * max(dx, dy))).to(temp.dtype)

    # mask of the cells where some extrapolation across phases occurred.
    extrapolated_cells = 1.0 #band_mask
    #(
    #    (lmask * ext_vapor_heat_flux != 0) | (vmask * ext_liquid_heat_flux != 0)
    #).to(temp.dtype) * band_mask
    
    return ext_liquid_heat_flux * extrapolated_cells, ext_vapor_heat_flux * extrapolated_cells

def mass_transfer(
    temp: torch.Tensor,
    sdf: torch.Tensor,
    sat_temp,
    dx: float,
    dy: float,
    stefan: float,
    reynolds: float,
    prandtl: float,
    thermal_conductivity: float,
    band_cells: int = DEFAULT_BAND_CELLS,
    taper_decay_cells: float = 2.0,
    wall_temp=None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Args:
        temp: cell-centered temperature, shape ``(..., H, W)``.
        sdf: cell-centered signed distance, shape ``(..., H, W)``. sdf < 0 is liquid,
            sdf >= 0 is vapor.
        sat_temp: interface (saturation) temperature; scalar or broadcastable to
            ``temp``.
        dx: cell spacing in x.
        dy: cell spacing in y.
        stefan: Stefan number.
        reynolds: Reynolds number.
        prandtl: Prandtl number.
        thermal_conductivity: vapor conductivity relative to the liquid.
        band_cells: half-width, in cells, of the interface band the flux is spread
            over (the extent over which the gradients are extrapolated).
        taper_decay_cells: decay length, in cells, of the exponential taper. ~2
            reproduces the Flash-X massflux decay (~0.6 per cell).
        wall_temp: heater temperature at the bottom wall, in the same units as
            ``temp``; applies a Dirichlet BC to the ghost-fluid gradients there.
            ``None`` keeps a zero-gradient wall.
        eps: floor passed through to the interface normals and ghost fluid
            gradients.

    Returns:
        Physical non-dimensional mass flux, shape ``(..., H, W)``: the Stefan jump
        on the ``band_cells``-wide band with an exponential taper, zero elsewhere.
    """
    # interface_heatflux already extrapolates each one-sided gradient across the
    # band, so both are defined on the band and the jump can be formed cell-wise.
    liquid_heatflux, vapor_heatflux = interface_heatflux(
        temp, sdf, sat_temp, dx, dy, band_cells, wall_temp, eps
    )
    conducted = liquid_heatflux - thermal_conductivity * vapor_heatflux
    return stefan / (reynolds * prandtl) * conducted


def continuity(
    temp: torch.Tensor,
    sdf: torch.Tensor,
    sat_temp,
    dx: float,
    dy: float,
    stefan: float,
    reynolds: float,
    prandtl: float,
    thermal_conductivity: float,
    rhogas: float,
    band_cells: int = DEFAULT_BAND_CELLS,
    taper_decay_cells: float = 2.0,
    wall_temp=None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Velocity-divergence source from phase change: ``mdot * (n . grad(rho))``.

    Args:
        temp: cell-centered temperature, shape ``(..., H, W)``.
        sdf: cell-centered signed distance, shape ``(..., H, W)``. ``sdf < 0`` is
            liquid, ``sdf >= 0`` is vapor.
        sat_temp: interface (saturation) temperature; scalar or broadcastable to
            ``temp``.
        dx: cell spacing in x.
        dy: cell spacing in y.
        stefan: Stefan number.
        reynolds: Reynolds number.
        prandtl: Prandtl number.
        thermal_conductivity: vapor conductivity relative to the liquid.
        rhogas: vapor-phase density.
        band_cells: half-width, in cells, of the interface band (see
            ``mass_transfer``).
        wall_temp: heater temperature at the bottom wall, in the same units as
            ``temp`` (see ``mass_transfer``). ``None`` keeps a zero-gradient wall.
        eps: floor passed through to the interface normals and ghost fluid
            gradients.

    Returns:
        Velocity-divergence source ``mdot * (n . grad(rho))``, shape ``(..., H, W)``,
        nonzero on the cells where ``grad(rho)`` is (straddling the interface).
    """
    mdot = mass_transfer(
        temp, sdf, sat_temp, dx, dy, stefan, reynolds, prandtl, thermal_conductivity,
        band_cells=band_cells, taper_decay_cells=taper_decay_cells, wall_temp=wall_temp, eps=eps,
    )

    normal_x, normal_y = interface_normals(sdf, dx, dy, eps)

    # Smear the density step over ~3 cells with a smoothed Heaviside of the SDF, so
    # grad(rho) spreads across a band instead of a single-cell spike. Liquid value
    # 1.0, vapor value 1/rhogas (heaviside runs 0 -> 1 from liquid to vapor).
    smear_cells = 5.0
    half_width = 0.5 * smear_cells * max(dx, dy)   # transition spans smear_cells cells
    phi = (sdf / half_width).clamp(-1.0, 1.0)
    heaviside = 0.5 * (1.0 + phi + torch.sin(torch.pi * phi) / torch.pi)
    rho = 1.0 + (1.0 / rhogas - 1.0) * heaviside
    grad_rho_y, grad_rho_x = torch.gradient(rho, spacing=(dy, dx), dim=(-2, -1), edge_order=1)

    return - mdot * (normal_x * grad_rho_x + normal_y * grad_rho_y)