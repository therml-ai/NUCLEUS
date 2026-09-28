import math

import torch

from interface_fields import circular_interface, vertical_interface
from nucleus.physics.mass_transfer import (
    DEFAULT_PROP_SMEAR,
    continuity,
    continuity_from_mass_flux,
    continuity_rhs,
    mass_transfer,
    stefan_mass_flux,
)
from nucleus.physics.phase_props import level_set_normals
from nucleus.physics.temp_grad import temp_gfm

FLUID = dict(stefan=1.0, reynolds=1.0, prandtl=1.0, thermal_conductivity=0.5, rhogas=0.01)


def _reference_loop_continuity_rhs(rhoc, normal_x, normal_y, mass_flux, dx, dy):
    """Index-by-index transcription of mph_evapDivergence2d.F90 in (H, W) layout."""
    height, width = rhoc.shape
    divergence = torch.zeros_like(rhoc)
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            rho_right = (rhoc[row, col] + rhoc[row, col + 1]) / 2.0
            rho_left = (rhoc[row, col] + rhoc[row, col - 1]) / 2.0
            rho_up = (rhoc[row, col] + rhoc[row + 1, col]) / 2.0
            rho_down = (rhoc[row, col] + rhoc[row - 1, col]) / 2.0
            flux_x = mass_flux[row, col] * normal_x[row, col]
            flux_y = mass_flux[row, col] * normal_y[row, col]
            divergence[row, col] = flux_x * (rho_right - rho_left) / dx + flux_y * (rho_up - rho_down) / dy
    return divergence


def test_continuity_rhs_matches_reference_loop():
    torch.manual_seed(3)
    height, width = 11, 14
    dx, dy = 0.09, 0.13
    rhoc = torch.rand(height, width, dtype=torch.float64) + 0.5
    mass_flux = torch.randn(height, width, dtype=torch.float64)
    normal_x = torch.randn(height, width, dtype=torch.float64)
    normal_y = torch.randn(height, width, dtype=torch.float64)

    divergence = continuity_rhs(rhoc, normal_x, normal_y, mass_flux, dx, dy)
    reference = _reference_loop_continuity_rhs(rhoc, normal_x, normal_y, mass_flux, dx, dy)
    assert torch.allclose(divergence, reference, atol=1e-10)
    assert torch.all(divergence[0, :] == 0) and torch.all(divergence[:, -1] == 0)


def test_stefan_mass_flux_on_planar_interface():
    height, width = 6, 40
    dx = dy = 0.05
    sat_temp, liquid_slope, vapor_slope = 1.0, 2.0, -0.7
    temp, sdf, x0 = vertical_interface(height, width, dx, dy, sat_temp, liquid_slope, vapor_slope)
    normal_x, normal_y = level_set_normals(sdf, dx, dy)
    liquid_flux, vapor_flux = temp_gfm(sdf, normal_x, normal_y, temp, sat_temp, dx, dy)

    mass_flux = stefan_mass_flux(liquid_flux, vapor_flux, 0.6, stefan=1.3, reynolds=1 / 0.8, prandtl=0.9)

    x = (torch.arange(width, dtype=torch.float64) + 0.5) * dx
    liquid_col = int((x < x0).sum().item()) - 1
    constant = 1.3 * 0.8 / 0.9
    rows = slice(1, -1)
    expected_liquid = torch.full((height - 2,), constant * liquid_slope, dtype=torch.float64)
    expected_vapor = torch.full((height - 2,), constant * (-0.6 * vapor_slope), dtype=torch.float64)
    assert torch.allclose(mass_flux[rows, liquid_col], expected_liquid, atol=1e-9)
    assert torch.allclose(mass_flux[rows, liquid_col + 1], expected_vapor, atol=1e-9)


def test_mass_transfer_can_be_either_sign():
    # mdot ~ liquid_slope - k * vapor_slope: negative (evaporation) when the liquid
    # is superheated, positive (condensation) when it is subcooled.
    height = width = 48
    dx = dy = 1.0 / 32
    temp, sdf, _ = vertical_interface(height, width, dx, dy, 1.0, liquid_slope=-10.0, vapor_slope=-2.0)
    evaporating = mass_transfer(temp, sdf, 1.0, dx, dy, **FLUID)
    assert (evaporating < 0).any() and not (evaporating > 0).any()

    temp, sdf, _ = vertical_interface(height, width, dx, dy, 1.0, liquid_slope=10.0, vapor_slope=2.0)
    condensing = mass_transfer(temp, sdf, 1.0, dx, dy, **FLUID)
    assert (condensing > 0).any() and not (condensing < 0).any()


def _support_reach_in_cells(field, sdf, dx):
    return (sdf.abs()[field != 0] / dx).max().item()


def test_mass_transfer_band_is_set_by_the_density_smear():
    # The normals come from the smoothed density and vanish where it is flat, so
    # the extrapolation cannot carry the flux past |sdf| ~ smear (plus the one-cell
    # stencil reach), however many iterations run. This is what keeps the band as
    # narrow as Flash-X's.
    height = width = 48
    dx = dy = 1.0 / 32
    temp, sdf, _ = vertical_interface(height, width, dx, dy, 1.0, -10.0, -2.0)

    reaches = []
    for prop_smear in (1.0, 3.0):
        mass_flux = mass_transfer(temp, sdf, 1.0, dx, dy, prop_smear=prop_smear, n_extrap_iters=8, **FLUID)
        reaches.append(_support_reach_in_cells(mass_flux, sdf, dx))
        assert reaches[-1] <= prop_smear * math.sqrt(2.0) + 1.0
    assert reaches[1] > reaches[0] + 1.0


def test_mass_transfer_batches_over_leading_dims_with_tensor_params():
    height = width = 32
    dx = dy = 1.0 / 32
    temp, sdf, _ = vertical_interface(height, width, dx, dy, 1.0, -10.0, -2.0)

    def batch(value):
        return torch.full((2, 3, 1, 1), value, dtype=torch.float64)

    tensor_fluid = {name: batch(value) for name, value in FLUID.items()}

    mass_flux = mass_transfer(
        temp.expand(2, 3, height, width), sdf.expand(2, 3, height, width), batch(1.0), dx, dy,
        wall_temp=batch(1.0), contact_angle=batch(45.0), **tensor_fluid,
    )
    single = mass_transfer(temp, sdf, 1.0, dx, dy, wall_temp=1.0, contact_angle=45.0, **FLUID)
    assert mass_flux.shape == (2, 3, height, width)
    assert torch.allclose(mass_flux[1, 2], single)


def test_continuity_is_finite_and_differentiable_on_a_bubble():
    torch.manual_seed(8)
    height, width = 40, 30
    dx = dy = 1.0 / 32
    sdf = circular_interface(height, width, dx, dy, center=(0.45, 0.5), radius=0.3)
    temp = (torch.rand(height, width, dtype=torch.float64) * 0.5).requires_grad_(True)

    divergence = continuity(temp, sdf, 0.0, dx, dy, wall_temp=1.0, contact_angle=45.0, **FLUID)
    assert divergence.shape == (height, width)
    assert torch.isfinite(divergence).all()
    assert (divergence != 0).any()
    # the source lives only where the smoothed density varies, near the interface
    assert _support_reach_in_cells(divergence, sdf, dx) <= DEFAULT_PROP_SMEAR * math.sqrt(2.0) + 1.0

    divergence.pow(2).sum().backward()
    assert temp.grad is not None and torch.isfinite(temp.grad).all()


def test_continuity_sign_matches_velocity_divergence():
    # An evaporating planar interface expels vapor: the specific volume jumps up
    # across the interface, so div(u) = -mdot * n.grad(1/rho) must be positive.
    height = width = 48
    dx = dy = 1.0 / 32
    temp, sdf, _ = vertical_interface(height, width, dx, dy, 1.0, -10.0, -2.0)

    divergence = continuity(temp, sdf, 1.0, dx, dy, **FLUID)
    assert (divergence > 0).any() and not (divergence < 0).any()


def test_continuity_from_mass_flux_matches_continuity_on_its_own_flux():
    height = width = 48
    dx = dy = 1.0 / 32
    temp, sdf, _ = vertical_interface(height, width, dx, dy, 1.0, -10.0, -2.0)
    kwargs = dict(wall_temp=1.0, contact_angle=45.0, **FLUID)

    mass_flux = mass_transfer(temp, sdf, 1.0, dx, dy, **kwargs)
    from_flux = continuity_from_mass_flux(mass_flux, sdf, dx, dy, FLUID["rhogas"], contact_angle=45.0)
    assert torch.allclose(from_flux, continuity(temp, sdf, 1.0, dx, dy, **kwargs), atol=1e-12)
