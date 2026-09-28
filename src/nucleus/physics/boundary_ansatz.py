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
    """Type of a single domain edge.
    ``NO_SLIP``: Dirichlet velocity wall. The ansatz decay vanishes on it, so the
                 prescribed wall value is enforced exactly. 
    ``OUTFLOW``: unconstrained edge. The decay stays one and no lift is applied, 
                 leaving the network free there.
    """
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
    """The decay factor ``g`` of the ansatz, evaluated on the given coordinates.

    ``g`` is a product of ``tanh`` ramps, one for each ``NO_SLIP`` edge: it is (to
    floating precision) one in the interior and decays smoothly to zero within roughly
    ``decay_length`` of every no-slip wall. ``OUTFLOW`` edges contribute no factor, so
    ``g`` stays one there and the ansatz leaves those edges unconstrained. Because
    ``tanh(0) = 0`` the decay is exactly zero on each no-slip wall, which is what makes
    ``V = B + g * NN`` satisfy the wall value regardless of ``NN``.

    Args:
        x_coords: x positions, any shape ``S``.
        y_coords: y positions, broadcastable to ``x_coords``.
        domain_width: x coordinate of the right wall.
        domain_height: y coordinate of the top edge.
        boundary_conditions: the type of each edge.
        decay_length: length scale over which ``g`` rises from 0 to ~1 off each no-slip
            wall.

    Returns:
        ``g`` in ``[0, 1]``, shape ``S`` (the broadcast of the inputs).
    """
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
    """Combine the pieces of the ansatz: ``V = B + g * NN``.

    ``boundary_lift`` and ``decay`` broadcast against ``network_output`` (they are the
    same on every batch/time element, so they usually carry only the trailing grid
    dimensions). Returns ``V`` with the shape of ``network_output``."""
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
    """Apply the boundary ansatz ``V = B + g * NN`` to a MAC velocity field.

    Builds the coordinate grids, decay ``g``, and homogeneous no-slip lift ``B`` for
    each velocity component and combines them with the network output. The x-velocity
    lives on the vertical faces ``(..., H, W + 1)`` and the y-velocity on the horizontal
    faces ``(..., H + 1, W)``; any leading (batch, time) dims broadcast. The grids are
    built on each component's device and dtype, and the lift is homogeneous, so the
    returned velocity vanishes on every no-slip wall regardless of the network.

    Args:
        model_velx: unconstrained x-velocity on the vertical faces, shape ``(..., H, W + 1)``.
        model_vely: unconstrained y-velocity on the horizontal faces, shape ``(..., H + 1, W)``.
        height: number of cells in y.
        width: number of cells in x.
        dx: cell spacing in x.
        dy: cell spacing in y.
        boundary_conditions: the type of each domain edge.
        decay_length_x: length scale of the x-velocity decay; defaults to ``4 * dx``.
        decay_length_y: length scale of the y-velocity decay; defaults to ``4 * dy``.

    Returns:
        ``(velx, vely)`` with the shapes of ``model_velx`` and ``model_vely``.
    """
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