import torch

from interface_fields import circular_interface, vertical_interface
from nucleus.physics.phase_props import level_set_normals
from nucleus.physics.temp_grad import _theta, temp_gfm


def _reference_loop_temp_gfm(phi, normal_x, normal_y, temp, sat_temp, dx, dy, tol):
    """Index-by-index transcription of mph_tempGfm2d.F90 in (H, W) = (y, x) layout,
    to cross-check the vectorized port against an independent translation."""
    height, width = phi.shape
    liquid_flux = torch.zeros_like(phi)
    vapor_flux = torch.zeros_like(phi)

    for row in range(1, height - 1):
        for col in range(1, width - 1):
            center = phi[row, col].item()
            temp_center = temp[row, col].item()
            neighbors = {
                "right": (row, col + 1), "left": (row, col - 1),
                "up": (row + 1, col), "down": (row - 1, col),
            }
            seen = {}
            for name, (nrow, ncol) in neighbors.items():
                neighbor = phi[nrow, ncol].item()
                if center * neighbor <= 0.0:
                    theta = abs(center) / (abs(center) + abs(neighbor))
                    seen[name] = temp_center + (sat_temp - temp_center) / max(tol, theta)
                else:
                    seen[name] = temp[nrow, ncol].item()

            grad_x = (seen["right"] - seen["left"]) / (2 * dx)
            grad_y = (seen["up"] - seen["down"]) / (2 * dy)
            projected = normal_x[row, col].item() * grad_x + normal_y[row, col].item() * grad_y
            vapor = 1.0 if center >= 0.0 else 0.0
            liquid_flux[row, col] = (1 - vapor) * projected
            vapor_flux[row, col] = vapor * (-projected)
    return liquid_flux, vapor_flux


def test_temp_gfm_matches_reference_loop():
    torch.manual_seed(0)
    height, width = 11, 14
    dx, dy = 0.09, 0.13
    sat_temp, tol = 0.25, 1e-2

    phi = -circular_interface(height, width, dx, dy, center=(0.6, 0.45), radius=0.35)
    temp = torch.rand(height, width, dtype=torch.float64) + 0.5
    normal_x, normal_y = level_set_normals(phi, dx, dy)

    liquid_flux, vapor_flux = temp_gfm(phi, normal_x, normal_y, temp, sat_temp, dx, dy, tol)
    liquid_ref, vapor_ref = _reference_loop_temp_gfm(phi, normal_x, normal_y, temp, sat_temp, dx, dy, tol)

    assert torch.allclose(liquid_flux, liquid_ref, atol=1e-10)
    assert torch.allclose(vapor_flux, vapor_ref, atol=1e-10)


def test_planar_stefan_condition_is_exact():
    # For a planar interface with a temperature that is piecewise linear in x, the
    # ghost-fluid extrapolation is exact: the liquid cell touching the interface must
    # see the liquid slope and the vapor cell the (negated) vapor slope.
    height, width = 6, 40
    dx = dy = 0.05
    sat_temp, liquid_slope, vapor_slope = 1.0, 2.0, -0.7
    temp, sdf, x0 = vertical_interface(height, width, dx, dy, sat_temp, liquid_slope, vapor_slope)
    normal_x, normal_y = level_set_normals(sdf, dx, dy)

    liquid_flux, vapor_flux = temp_gfm(sdf, normal_x, normal_y, temp, sat_temp, dx, dy)

    x = (torch.arange(width, dtype=torch.float64) + 0.5) * dx
    liquid_col = int((x < x0).sum().item()) - 1
    vapor_col = liquid_col + 1
    rows = slice(1, -1)
    expected_liquid = torch.full((height - 2,), liquid_slope, dtype=torch.float64)
    expected_vapor = torch.full((height - 2,), -vapor_slope, dtype=torch.float64)
    assert torch.allclose(liquid_flux[rows, liquid_col], expected_liquid, atol=1e-10)
    assert torch.allclose(vapor_flux[rows, vapor_col], expected_vapor, atol=1e-10)
    # and each is zero on the other phase
    assert torch.all(vapor_flux[rows, liquid_col] == 0)
    assert torch.all(liquid_flux[rows, vapor_col] == 0)


def test_temp_gfm_degenerate_phi_stays_finite_forwards_and_backwards():
    # With phi == 0 in adjacent cells the Fortran theta is 0/0, but gfortran's
    # max(tol, NaN) returns tol, so the result is finite there -- and so must the
    # gradient be, since torch.where leaks NaN from the unselected branch.
    height, width = 6, 6
    phi = torch.zeros(height, width, dtype=torch.float64)
    temp = torch.rand(height, width, dtype=torch.float64).requires_grad_(True)
    normal_x = torch.ones(height, width, dtype=torch.float64)
    normal_y = torch.zeros(height, width, dtype=torch.float64)

    liquid_flux, vapor_flux = temp_gfm(phi, normal_x, normal_y, temp, 0.5, 0.1, 0.1)
    assert torch.isfinite(liquid_flux).all() and torch.isfinite(vapor_flux).all()

    (liquid_flux.sum() + vapor_flux.sum()).backward()
    assert torch.isfinite(temp.grad).all()

    zeros = torch.zeros(3, dtype=torch.float64)
    assert torch.allclose(_theta(zeros, zeros, 1e-2), torch.full((3,), 1e-2, dtype=torch.float64))


def test_temp_gfm_batches_over_leading_dims():
    height, width = 8, 12
    dx = dy = 0.1
    temp, sdf, _ = vertical_interface(height, width, dx, dy, 1.0, 2.0, -5.0)
    normal_x, normal_y = level_set_normals(sdf, dx, dy)

    def expand(field):
        return field.expand(2, 3, height, width)

    liquid_flux, vapor_flux = temp_gfm(
        expand(sdf), expand(normal_x), expand(normal_y), expand(temp), 1.0, dx, dy
    )
    single_liquid, single_vapor = temp_gfm(sdf, normal_x, normal_y, temp, 1.0, dx, dy)
    assert liquid_flux.shape == (2, 3, height, width)
    assert torch.allclose(liquid_flux[1, 2], single_liquid)
    assert torch.allclose(vapor_flux[1, 2], single_vapor)
