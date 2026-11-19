import math

import torch

from interface_fields import vertical_interface
from nucleus.physics.phase_props import (
    level_set_normals,
    min_cell_diag,
    smooth_density,
    smoothed_heaviside,
)


def _reference_loop_smoothed_heaviside(phi, smear):
    """Index-by-index transcription of Stencils_lsCenterPropsSmeared's smhv formula."""
    out = torch.empty_like(phi)
    for index in range(phi.numel()):
        value = phi.reshape(-1)[index].item()
        if abs(value) <= smear:
            out.reshape(-1)[index] = 0.5 + value / (2 * smear) + math.sin(2 * math.pi * value / (2 * smear)) / (2 * math.pi)
        else:
            out.reshape(-1)[index] = 1.0 if value >= 0.0 else 0.0
    return out


def test_smoothed_heaviside_matches_reference_loop_and_limits():
    torch.manual_seed(5)
    smear = 0.37
    phi = (torch.rand(50, dtype=torch.float64) - 0.5) * 4 * smear

    assert torch.allclose(smoothed_heaviside(phi, smear), _reference_loop_smoothed_heaviside(phi, smear), atol=1e-12)

    limits = torch.tensor([0.0, 10 * smear, -10 * smear], dtype=torch.float64)
    assert torch.allclose(smoothed_heaviside(limits, smear), torch.tensor([0.5, 1.0, 0.0], dtype=torch.float64))


def test_smooth_density_blends_to_pure_phases_away_from_interface():
    smear = 0.1
    rho_gas, rho_liquid = 200.0, 1.0
    phi = torch.tensor([-1.0, -0.05, 0.0, 0.05, 1.0], dtype=torch.float64)

    rhoc, smhv = smooth_density(phi, rho_gas, rho_liquid, smear)
    assert torch.isclose(rhoc[0], torch.tensor(rho_liquid, dtype=torch.float64))
    assert torch.isclose(rhoc[-1], torch.tensor(rho_gas, dtype=torch.float64))
    assert torch.isclose(rhoc[2], torch.tensor(0.5 * (rho_gas + rho_liquid), dtype=torch.float64))
    assert torch.all(rhoc >= rho_liquid - 1e-12) and torch.all(rhoc <= rho_gas + 1e-12)
    assert torch.allclose(smhv, smoothed_heaviside(phi, smear))


def test_smooth_density_broadcasts_batched_gas_density():
    phi = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64).expand(2, 3, 9)
    rho_gas = torch.tensor([50.0, 200.0], dtype=torch.float64)[:, None, None]

    rhoc, _ = smooth_density(phi, rho_gas, 1.0, 0.3)
    assert rhoc.shape == (2, 3, 9)
    assert torch.isclose(rhoc[0, 0, -1], torch.tensor(50.0, dtype=torch.float64))
    assert torch.isclose(rhoc[1, 0, -1], torch.tensor(200.0, dtype=torch.float64))


def test_min_cell_diag():
    assert math.isclose(min_cell_diag(3.0, 4.0), 5.0)


def test_normals_of_vertical_interface_point_along_x():
    height, width = 12, 16
    dx = dy = 0.1
    _, sdf, _ = vertical_interface(height, width, dx, dy, 1.0, 1.0, 1.0)

    normal_x, normal_y = level_set_normals(sdf, dx, dy)
    interior = (slice(1, -1), slice(1, -1))
    assert torch.allclose(normal_x[interior], torch.ones_like(normal_x[interior]), atol=1e-6)
    assert torch.allclose(normal_y[interior], torch.zeros_like(normal_y[interior]), atol=1e-6)
    # the 1-cell border is left at zero, as in the Fortran
    assert torch.all(normal_x[0, :] == 0) and torch.all(normal_x[:, -1] == 0)


def test_normals_from_specific_volume_point_liquid_to_vapor():
    # rhoc is 1/rho, increasing into the vapor, so its normals agree with grad(sdf);
    # blending plain densities instead would flip them.
    height, width = 12, 16
    dx = dy = 0.1
    _, sdf, _ = vertical_interface(height, width, dx, dy, 1.0, 1.0, 1.0)
    rhoc, _ = smooth_density(sdf, 1.0 / 0.005, 1.0, 1.5 * min_cell_diag(dx, dy))

    normal_x, _ = level_set_normals(rhoc, dx, dy)
    live = normal_x.abs() > 1e-6
    assert live.any()
    assert torch.all(normal_x[live] > 0)
