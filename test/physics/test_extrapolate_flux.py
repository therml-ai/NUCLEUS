import torch

from nucleus.physics.extrapolate_flux import extrapolate_phase_flux
from nucleus.physics.sdf import interface_normals


def _vertical_sdf(height, width, dx, dy):
    """Signed distance for a vertical interface: liquid (sdf < 0) left, vapor right.
    The normal is +x everywhere."""
    x = (torch.arange(width, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(height, dtype=torch.float64) + 0.5) * dy
    _, grid_x = torch.meshgrid(y, x, indexing="ij")
    x0 = x[width // 2] + 0.5 * dx
    return grid_x - x0


def test_extrapolate_phase_flux_leaves_a_globally_constant_field_unchanged():
    # A field that's already the same constant on both phases is an exact fixed
    # point of the march: n.grad(field) = 0 everywhere, so nothing should move.
    dx = dy = 1.0 / 32
    H = W = 64
    sdf = _vertical_sdf(H, W, dx, dy)
    normal_x, normal_y = interface_normals(sdf, dx, dy)

    q_l = torch.full_like(sdf, 2.0)
    q_v = torch.full_like(sdf, -3.0)

    ext_q_l, ext_q_v = extrapolate_phase_flux(q_l, q_v, sdf, normal_x, normal_y, dx, dy)

    assert torch.equal(ext_q_l, q_l)
    assert torch.equal(ext_q_v, q_v)


def test_extrapolate_phase_flux_preserves_source_phase_values():
    # q_l is only meaningful on the liquid phase (its source); q_v only on vapor.
    # Each is only overwritten on the *opposite* phase, so the source-phase values
    # must survive exactly.
    dx = dy = 1.0 / 32
    H = W = 48
    sdf = _vertical_sdf(H, W, dx, dy)
    normal_x, normal_y = interface_normals(sdf, dx, dy)

    q_l = torch.where(sdf < 0, torch.full_like(sdf, 5.0), torch.zeros_like(sdf))
    q_v = torch.where(sdf >= 0, torch.full_like(sdf, -7.0), torch.zeros_like(sdf))

    ext_q_l, ext_q_v = extrapolate_phase_flux(q_l, q_v, sdf, normal_x, normal_y, dx, dy)

    assert torch.all(ext_q_l[sdf < 0] == q_l[sdf < 0])
    assert torch.all(ext_q_v[sdf >= 0] == q_v[sdf >= 0])
