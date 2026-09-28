import torch

from interface_fields import vertical_interface
from nucleus.physics.extrapolate_flux import (
    advect_upwind_rhs,
    extrapolate_phase_fluxes,
    phased_update,
)
from nucleus.physics.guard_cells import refill_guard_cells


def _reference_loop_advect_upwind(field, vel_x, vel_y, dx, dy):
    """Index-by-index transcription of Stencils_cnt_advectUpwind2d.F90 in (H, W)
    layout, to cross-check the vectorized port against an independent translation."""
    height, width = field.shape
    rhs = torch.zeros_like(field)
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            u_conv, v_conv = vel_x[row, col].item(), vel_y[row, col].item()
            forward_x = field[row, col + 1].item() - field[row, col].item()
            backward_x = field[row, col].item() - field[row, col - 1].item()
            forward_y = field[row + 1, col].item() - field[row, col].item()
            backward_y = field[row, col].item() - field[row - 1, col].item()
            rhs[row, col] = (
                -(max(u_conv, 0.0) * backward_x + min(u_conv, 0.0) * forward_x) / dx
                - (max(v_conv, 0.0) * backward_y + min(v_conv, 0.0) * forward_y) / dy
            )
    return rhs


def test_advect_upwind_rhs_matches_reference_loop():
    torch.manual_seed(6)
    height, width = 10, 13
    dx, dy = 0.11, 0.08
    field = torch.randn(height, width, dtype=torch.float64)
    vel_x = torch.randn(height, width, dtype=torch.float64)
    vel_y = torch.randn(height, width, dtype=torch.float64)

    rhs = advect_upwind_rhs(field, vel_x, vel_y, dx, dy)
    assert torch.allclose(rhs, _reference_loop_advect_upwind(field, vel_x, vel_y, dx, dy), atol=1e-10)
    assert torch.all(rhs[0, :] == 0) and torch.all(rhs[:, -1] == 0)


def test_phased_update_only_touches_the_vapor_side():
    phi = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0], dtype=torch.float64)
    field = torch.full((5,), 2.0, dtype=torch.float64)
    rhs = torch.full((5,), 3.0, dtype=torch.float64)

    out = phased_update(field, rhs, phi, dt=0.5)
    assert torch.allclose(out, torch.tensor([2.0, 2.0, 3.5, 3.5, 3.5], dtype=torch.float64))


def test_extrapolation_advances_one_cell_per_iteration():
    # On a planar interface with a constant +x normal the scheme is 1-cell-per-
    # iteration upwind: the liquid flux seeded at the interface-adjacent liquid cell
    # must be exactly zero beyond n_iterations cells into the vapor (and vice versa
    # for the vapor flux), and the seed side is untouched.
    height, width = 4, 30
    dx = dy = 0.1
    n_iterations = 4
    _, sdf, x0 = vertical_interface(height, width, dx, dy, 1.0, 1.0, 1.0)
    normal_x = torch.ones_like(sdf)
    normal_y = torch.zeros_like(sdf)

    x = (torch.arange(width, dtype=torch.float64) + 0.5) * dx
    liquid_col = int((x < x0).sum().item()) - 1
    vapor_col = liquid_col + 1
    liquid_flux = torch.zeros_like(sdf)
    liquid_flux[:, liquid_col] = 1.0
    vapor_flux = torch.zeros_like(sdf)
    vapor_flux[:, vapor_col] = 1.0

    liquid_ext, vapor_ext = extrapolate_phase_fluxes(
        sdf, normal_x, normal_y, liquid_flux, vapor_flux, dx, dy, n_iterations
    )
    assert torch.all(liquid_ext[:, liquid_col + n_iterations + 1:-1] == 0)
    assert torch.all(vapor_ext[:, 1:vapor_col - n_iterations] == 0)
    assert torch.all(liquid_ext[:, 1:liquid_col] == 0)
    assert torch.all(vapor_ext[:, vapor_col + 1:-1] == 0)
    # and the front has actually moved
    assert torch.all(liquid_ext[1:-1, liquid_col + 1:liquid_col + n_iterations + 1] > 0)


def test_guard_refill_keeps_a_wall_flux_out_of_the_domain():
    # A halo built by reflecting temperature about a Dirichlet wall produces a large
    # fictitious one-sided flux under the wall; without refilling the guard cells
    # every iteration the extrapolation advects it back into the domain.
    n_guard, dx = 3, 1.0 / 32.0
    # the wall is the low-y edge (row 0) and the normals point away from it, +y
    wall_distance = torch.arange(20, dtype=torch.float64).reshape(-1, 1) - 1.0
    phi = (wall_distance * dx).expand(20, 20).contiguous()
    liquid_flux = torch.ones(20, 20, dtype=torch.float64)
    liquid_flux[:n_guard] = 50.0
    normal_x = torch.zeros_like(phi)
    normal_y = torch.ones_like(phi)

    raw, _ = extrapolate_phase_fluxes(phi, normal_x, normal_y, liquid_flux, liquid_flux.clone(), dx, dx, 5)
    filled, _ = extrapolate_phase_fluxes(
        phi, normal_x, normal_y, liquid_flux, liquid_flux.clone(), dx, dx, 5, n_guard=n_guard
    )
    interior = slice(n_guard, -n_guard)
    assert raw[interior, interior].max() > 10.0
    assert filled[interior, interior].max() == 1.0
    assert torch.equal(filled, refill_guard_cells(filled, n_guard))
