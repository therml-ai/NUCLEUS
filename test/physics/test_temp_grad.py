import torch

from nucleus.physics.temp_grad import vapor_temp_grad, liquid_temp_grad


def _vertical_interface(height, width, dx, dy, x0, sat_temp, vapor_slope, liquid_slope):
    """A vertical interface at x0 with each phase linear in x, meeting the
    saturation temperature at the interface with its own (different) slope."""
    x = (torch.arange(width, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(height, dtype=torch.float64) + 0.5) * dy
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    sdf = grid_x - x0                      # vapor where x > x0 (sdf >= 0)
    temp = torch.where(
        sdf >= 0,
        sat_temp + vapor_slope * (grid_x - x0),
        sat_temp + liquid_slope * (grid_x - x0),
    )
    return temp, sdf


def test_ghost_fluid_recovers_each_phase_slope():
    # Each phase is linear with a different slope; the ghost fill must recover the
    # phase's own slope right up to the interface, not the smeared central value.
    H = W = 64
    dx = dy = 1.0 / 32
    sat_temp, vapor_slope, liquid_slope = 1.0, 2.0, -5.0
    x = (torch.arange(W, dtype=torch.float64) + 0.5) * dx
    x0 = x[W // 2] + 0.5 * dx               # off a cell center -> no cell has sdf == 0

    temp, sdf = _vertical_interface(H, W, dx, dy, x0, sat_temp, vapor_slope, liquid_slope)
    vapor_gx, vapor_gy = vapor_temp_grad(temp, sdf, sat_temp, dx, dy)
    liquid_gx, liquid_gy = liquid_temp_grad(temp, sdf, sat_temp, dx, dy)

    vapor, liquid = sdf >= 0, sdf < 0
    # interface-adjacent cells, where a naive central difference would smear
    vapor_edge = vapor & (sdf < 1.5 * dx)
    liquid_edge = liquid & (sdf > -1.5 * dx)

    assert torch.allclose(vapor_gx[vapor_edge], torch.tensor(vapor_slope, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(liquid_gx[liquid_edge], torch.tensor(liquid_slope, dtype=torch.float64), atol=1e-9)

    # no cross-derivative for an x-only field
    assert vapor_gy[vapor].abs().max() < 1e-9
    assert liquid_gy[liquid].abs().max() < 1e-9

    # and a plain central difference really is smeared at the interface (so the
    # ghost fill is doing something)
    naive = torch.gradient(temp, spacing=dx, dim=-1)[0]
    assert (naive[vapor_edge] - vapor_slope).abs().max() > 1.0


def test_gradients_are_masked_to_their_phase():
    H = W = 48
    dx = dy = 1.0 / 32
    x = (torch.arange(W, dtype=torch.float64) + 0.5) * dx
    x0 = x[W // 2] + 0.5 * dx
    temp, sdf = _vertical_interface(H, W, dx, dy, x0, 1.0, 2.0, -5.0)

    vapor_gx, vapor_gy = vapor_temp_grad(temp, sdf, 1.0, dx, dy)
    liquid_gx, liquid_gy = liquid_temp_grad(temp, sdf, 1.0, dx, dy)

    assert torch.all(vapor_gx[sdf < 0] == 0) and torch.all(vapor_gy[sdf < 0] == 0)
    assert torch.all(liquid_gx[sdf >= 0] == 0) and torch.all(liquid_gy[sdf >= 0] == 0)


def test_wall_dirichlet_recovers_normal_gradient_at_bottom():
    # An all-liquid column with a linear vertical profile T = wall + slope*y. The
    # Dirichlet wall BC must recover the true slope at the bottom row (index 0),
    # where a replicate pad instead halves it (ghost below = row 0 -> (T1-T0)/2dy).
    H = W = 16
    dx = dy = 1.0 / 32
    slope, wall = 3.0, 0.5
    y = (torch.arange(H, dtype=torch.float64) + 0.5) * dy
    temp = (wall + slope * y).view(H, 1).expand(H, W).contiguous()
    sdf = torch.full((H, W), -1.0, dtype=torch.float64)   # all liquid, no interface

    _, replicate_gy = liquid_temp_grad(temp, sdf, sat_temp=0.0, dx=dx, dy=dy)
    _, wall_gy = liquid_temp_grad(temp, sdf, sat_temp=0.0, dx=dx, dy=dy, wall_temp=wall)

    center = W // 2
    assert abs(float(replicate_gy[0, center]) - slope / 2) < 1e-9   # replicate halves it
    assert abs(float(wall_gy[0, center]) - slope) < 1e-9            # Dirichlet recovers it
    # interior rows are identical under both (the BC only touches the bottom row)
    assert torch.allclose(replicate_gy[2:], wall_gy[2:])


def test_temp_grad_batches_over_leading_dims():
    H = W = 32
    dx = dy = 1.0 / 32
    x = (torch.arange(W, dtype=torch.float64) + 0.5) * dx
    x0 = x[W // 2] + 0.5 * dx
    temp, sdf = _vertical_interface(H, W, dx, dy, x0, 1.0, 2.0, -5.0)

    batch_temp = temp.expand(2, 3, H, W)
    batch_sdf = sdf.expand(2, 3, H, W)
    grad_x, grad_y = vapor_temp_grad(batch_temp, batch_sdf, 1.0, dx, dy)
    assert grad_x.shape == (2, 3, H, W)
    # each slice matches the single-frame result
    single_x, _ = vapor_temp_grad(temp, sdf, 1.0, dx, dy)
    assert torch.allclose(grad_x[1, 2], single_x)
