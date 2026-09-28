from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch

from nucleus.physics.coordinates import (
    domain_extent,
    x_face_coordinates,
    y_face_coordinates,
)
from nucleus.physics.poisson import GRID_SPACING


class BoundaryType(Enum):
    """``NO_SLIP`` walls get the exact prescribed value; ``OUTFLOW`` edges are left
    to the network."""
    NO_SLIP = auto()
    OUTFLOW = auto()


@dataclass(frozen=True)
class BoundaryConditions:
    left: BoundaryType = BoundaryType.NO_SLIP
    right: BoundaryType = BoundaryType.NO_SLIP
    bottom: BoundaryType = BoundaryType.NO_SLIP
    top: BoundaryType = BoundaryType.OUTFLOW


def boundary_decay(
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    domain_width: float,
    domain_height: float,
    boundary_conditions: BoundaryConditions,
    decay_length: float = 4.0 * GRID_SPACING,
) -> torch.Tensor:
    """Decay ``g`` of the ansatz ``V = B + g * NN``: a product of ``tanh`` ramps, one
    per ``NO_SLIP`` edge, so ``g`` is exactly zero on each no-slip wall and ~1 a few
    ``decay_length`` away. Shape is the broadcast of ``x_coords`` and ``y_coords``."""
    decay = torch.ones_like(x_coords + y_coords)
    if boundary_conditions.left is BoundaryType.NO_SLIP:
        decay = decay * torch.tanh(x_coords / decay_length)
    if boundary_conditions.right is BoundaryType.NO_SLIP:
        decay = decay * torch.tanh((domain_width - x_coords) / decay_length)
    if boundary_conditions.bottom is BoundaryType.NO_SLIP:
        decay = decay * torch.tanh(y_coords / decay_length)
    if boundary_conditions.top is BoundaryType.NO_SLIP:
        decay = decay * torch.tanh((domain_height - y_coords) / decay_length)
    return decay


def boundary_lift(
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    domain_width: float,
    domain_height: float,
    boundary_conditions: BoundaryConditions = BoundaryConditions(),
    left_value: float = 0.0,
    right_value: float = 0.0,
    bottom_value: float = 0.0,
    top_value: float = 0.0,
    decay_length: float = 4.0 * GRID_SPACING,
) -> torch.Tensor:
    lift = torch.zeros_like(x_coords + y_coords)
    if boundary_conditions.left is BoundaryType.NO_SLIP and left_value != 0.0:
        lift = lift + left_value * (1.0 - torch.tanh(x_coords / decay_length))
    if boundary_conditions.right is BoundaryType.NO_SLIP and right_value != 0.0:
        lift = lift + right_value * (1.0 - torch.tanh((domain_width - x_coords) / decay_length))
    if boundary_conditions.bottom is BoundaryType.NO_SLIP and bottom_value != 0.0:
        lift = lift + bottom_value * (1.0 - torch.tanh(y_coords / decay_length))
    if boundary_conditions.top is BoundaryType.NO_SLIP and top_value != 0.0:
        lift = lift + top_value * (1.0 - torch.tanh((domain_height - y_coords) / decay_length))
    return lift


def apply_ansatz(
    network_output: torch.Tensor, boundary_lift: torch.Tensor, decay: torch.Tensor
) -> torch.Tensor:
    """``V = B + g * NN``; ``boundary_lift`` and ``decay`` broadcast against
    ``network_output``."""
    return boundary_lift + decay * network_output


def vel_ansatz(
    model_velx: torch.Tensor,
    model_vely: torch.Tensor,
    height: int,
    width: int,
    dx: float,
    dy: float,
    boundary_conditions: BoundaryConditions,
    decay_length_x: Optional[float] = None,
    decay_length_y: Optional[float] = None,
):
    """Boundary ansatz ``V = B + g * NN`` on a MAC velocity: ``model_velx`` is
    ``(..., H, W + 1)``, ``model_vely`` is ``(..., H + 1, W)``, and the returned
    ``(velx, vely)`` match those shapes and vanish on every no-slip wall. Decay
    lengths default to four cells."""
    if decay_length_x is None:
        decay_length_x = 4 * dx
    if decay_length_y is None:
        decay_length_y = 4 * dy

    domain_width, domain_height = domain_extent(height, width, dx, dy)

    face_x_x, face_x_y = x_face_coordinates(
        height, width, dx, dy, model_velx.device, model_velx.dtype
    )
    decay_x = boundary_decay(
        face_x_x, face_x_y, domain_width, domain_height, boundary_conditions, decay_length_x
    )
    lift_x = boundary_lift(
        face_x_x, face_x_y, domain_width, domain_height, boundary_conditions, decay_length=decay_length_x
    )
    velx = apply_ansatz(model_velx, lift_x, decay_x)

    face_y_x, face_y_y = y_face_coordinates(
        height, width, dx, dy, model_vely.device, model_vely.dtype
    )
    decay_y = boundary_decay(
        face_y_x, face_y_y, domain_width, domain_height, boundary_conditions, decay_length_y
    )
    lift_y = boundary_lift(
        face_y_x, face_y_y, domain_width, domain_height, boundary_conditions, decay_length=decay_length_y
    )
    vely = apply_ansatz(model_vely, lift_y, decay_y)

    return velx, vely