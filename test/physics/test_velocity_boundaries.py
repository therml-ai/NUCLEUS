import pytest
import torch

from nucleus.physics.poisson import divergence_centers_from_faces
from nucleus.physics.velocity_boundaries import FaceVelocityBoundary, apply_velocity_boundaries

DX, DY = 0.1, 0.2
NONE = FaceVelocityBoundary.NONE
WALL = FaceVelocityBoundary.WALL
NEUMANN = FaceVelocityBoundary.NEUMANN
CONTINUITY = FaceVelocityBoundary.CONTINUITY


def random_faces(height: int = 6, width: int = 5, batch: int = 2):
    generator = torch.Generator().manual_seed(0)
    velfacex = torch.randn(batch, height, width + 1, generator=generator, dtype=torch.float64)
    velfacey = torch.randn(batch, height + 1, width, generator=generator, dtype=torch.float64)
    return velfacex, velfacey


def apply(velfacex, velfacey, left=NONE, right=NONE, bottom=NONE, top=NONE):
    return apply_velocity_boundaries(
        velfacex, velfacey, DX, DY, left=left, right=right, bottom=bottom, top=top
    )


def test_none_leaves_the_faces_unchanged_and_copies():
    velfacex, velfacey = random_faces()
    new_velfacex, new_velfacey = apply(velfacex, velfacey)
    assert torch.equal(new_velfacex, velfacex) and torch.equal(new_velfacey, velfacey)
    assert new_velfacex.data_ptr() != velfacex.data_ptr()


def test_walls_zero_only_the_normal_boundary_faces():
    velfacex, velfacey = random_faces()
    new_velfacex, new_velfacey = apply(velfacex, velfacey, left=WALL, right=WALL, bottom=WALL)
    assert (new_velfacex[..., 0] == 0).all() and (new_velfacex[..., -1] == 0).all()
    assert (new_velfacey[..., 0, :] == 0).all()
    assert torch.equal(new_velfacex[..., 1:-1], velfacex[..., 1:-1])
    assert torch.equal(new_velfacey[..., 1:, :], velfacey[..., 1:, :])


def test_neumann_copies_the_adjacent_face():
    velfacex, velfacey = random_faces()
    new_velfacex, new_velfacey = apply(velfacex, velfacey, right=NEUMANN, top=NEUMANN)
    assert torch.equal(new_velfacex[..., -1], velfacex[..., -2])
    assert torch.equal(new_velfacey[..., -1, :], velfacey[..., -2, :])


def test_continuity_makes_every_boundary_cell_divergence_free():
    velfacex, velfacey = random_faces()
    new_velfacex, new_velfacey = apply(
        velfacex, velfacey, left=CONTINUITY, right=CONTINUITY, bottom=CONTINUITY, top=CONTINUITY
    )
    divergence = divergence_centers_from_faces(new_velfacex, new_velfacey, DX, DY)
    for boundary_cells in (divergence[..., 0, :], divergence[..., -1, :], divergence[..., :, 0], divergence[..., :, -1]):
        assert torch.allclose(boundary_cells, torch.zeros_like(boundary_cells), atol=1e-12)


def test_walls_with_continuity_outflow_match_a_flashx_style_field():
    velfacex, velfacey = random_faces()
    new_velfacex, new_velfacey = apply(velfacex, velfacey, left=WALL, right=WALL, bottom=WALL, top=CONTINUITY)
    top_cells = divergence_centers_from_faces(new_velfacex, new_velfacey, DX, DY)[..., -1, :]
    assert torch.allclose(top_cells, torch.zeros_like(top_cells), atol=1e-12)
    assert (new_velfacex[..., 0] == 0).all() and (new_velfacey[..., 0, :] == 0).all()


def test_gradients_flow_through_the_boundary_conditions():
    velfacex, velfacey = random_faces()
    velfacex.requires_grad_(True)
    velfacey.requires_grad_(True)
    new_velfacex, new_velfacey = apply(velfacex, velfacey, left=WALL, right=NEUMANN, top=CONTINUITY)
    (new_velfacex.sum() + new_velfacey.sum()).backward()
    assert (velfacex.grad[..., 0] == 0).all()
    assert velfacey.grad.abs().sum() > 0


def test_unknown_condition_is_rejected():
    velfacex, velfacey = random_faces()
    with pytest.raises(ValueError, match="unknown velocity boundary"):
        apply(velfacex, velfacey, left="wall")
