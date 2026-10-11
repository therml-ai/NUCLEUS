from enum import Enum, auto
from typing import Tuple

import torch


class FaceVelocityBoundary(Enum):
    """How the normal velocity on a boundary's faces is set.

    ``CONTINUITY`` makes the adjacent row/column of cells divergence-free, which is
    how Flash-X sets its outflow faces; ``NEUMANN`` copies the adjacent interior face.
    """
    NONE = auto()
    WALL = auto()
    NEUMANN = auto()
    CONTINUITY = auto()


def apply_velocity_boundaries(
    velfacex: torch.Tensor,
    velfacey: torch.Tensor,
    dx: float,
    dy: float,
    left: FaceVelocityBoundary,
    right: FaceVelocityBoundary,
    bottom: FaceVelocityBoundary,
    top: FaceVelocityBoundary,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Set the normal face velocity on each boundary of ``velfacex (..., H, W+1)`` and
    ``velfacey (..., H+1, W)``; tangential faces are never on a boundary."""
    velfacex, velfacey = velfacex.clone(), velfacey.clone()
    # x boundaries first, so a CONTINUITY y boundary sees the final x faces and its
    # corner cells end up divergence-free
    _apply_side(velfacex, velfacey, dx, dy, dim=-1, index=0, condition=left)
    _apply_side(velfacex, velfacey, dx, dy, dim=-1, index=-1, condition=right)
    _apply_side(velfacey, velfacex, dy, dx, dim=-2, index=0, condition=bottom)
    _apply_side(velfacey, velfacex, dy, dx, dim=-2, index=-1, condition=top)
    return velfacex, velfacey


def _apply_side(
    normal: torch.Tensor,
    tangential: torch.Tensor,
    normal_spacing: float,
    tangential_spacing: float,
    dim: int,
    index: int,
    condition: FaceVelocityBoundary,
) -> None:
    boundary = normal.select(dim, index)
    neighbor = normal.select(dim, 1 if index == 0 else -2)
    if condition is FaceVelocityBoundary.NONE:
        return
    if condition is FaceVelocityBoundary.WALL:
        boundary.zero_()
    elif condition is FaceVelocityBoundary.NEUMANN:
        boundary.copy_(neighbor)
    elif condition is FaceVelocityBoundary.CONTINUITY:
        boundary.copy_(_divergence_free_face(neighbor, tangential, normal_spacing, tangential_spacing, dim, index))
    else:
        raise ValueError(f"unknown velocity boundary condition {condition!r}")


def _divergence_free_face(
    neighbor: torch.Tensor,
    tangential: torch.Tensor,
    normal_spacing: float,
    tangential_spacing: float,
    dim: int,
    index: int,
) -> torch.Tensor:
    # the boundary cells' tangential faces, selected along the same axis, leave the
    # cells' own axis last, so the tangential derivative is a diff over it
    boundary_cells = tangential.select(dim, index)
    tangential_derivative = torch.diff(boundary_cells, dim=-1) / tangential_spacing
    # zero divergence: (high face - low face) / normal_spacing + tangential_derivative = 0
    direction = 1.0 if index == 0 else -1.0
    return neighbor + direction * normal_spacing * tangential_derivative
