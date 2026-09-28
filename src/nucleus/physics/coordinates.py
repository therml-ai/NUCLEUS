"""Physical coordinate grids for the staggered MAC layout used throughout
``nucleus.physics``.

The domain has its left and bottom walls at the origin, the right wall at
``x = width * dx``, and the top outflow at ``y = height * dy``. Cell centers sit at
``((i + 0.5) * dx, (j + 0.5) * dy)``; MAC faces sit on the cell edges so a velocity
face can land exactly on a wall.
"""

from typing import Tuple

import torch


def domain_extent(height: int, width: int, dx: float, dy: float) -> Tuple[float, float]:
    """Physical size ``(domain_width, domain_height)`` of a ``height`` x ``width`` cell
    grid: the right wall sits at ``x = width * dx`` and the top outflow at
    ``y = height * dy``, with the left/bottom walls at the origin."""
    return width * dx, height * dy


def _meshgrid(x_axis: torch.Tensor, y_axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(x_coords, y_coords)`` for the tensor product of ``x_axis`` and ``y_axis``,
    each shape ``(len(y_axis), len(x_axis))`` with x on the last (width) axis."""
    y_coords, x_coords = torch.meshgrid(y_axis, x_axis, indexing="ij")
    return x_coords, y_coords


def x_face_coordinates(
    height: int, width: int, dx: float, dy: float, device=None, dtype=torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Physical ``(x, y)`` coordinates of the x-velocity faces, each shape
    ``(H, W + 1)``. The faces sit at ``x = i * dx`` (``i = 0..W``, so exactly on the
    left and right walls) and at the cell-center heights ``y = (j + 0.5) * dy``."""
    x_axis = torch.arange(width + 1, device=device, dtype=dtype) * dx
    y_axis = (torch.arange(height, device=device, dtype=dtype) + 0.5) * dy
    return _meshgrid(x_axis, y_axis)


def y_face_coordinates(
    height: int, width: int, dx: float, dy: float, device=None, dtype=torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Physical ``(x, y)`` coordinates of the y-velocity faces, each shape
    ``(H + 1, W)``. The faces sit at the cell-center widths ``x = (i + 0.5) * dx`` and
    at ``y = j * dy`` (``j = 0..H``, so exactly on the bottom wall and top outflow)."""
    x_axis = (torch.arange(width, device=device, dtype=dtype) + 0.5) * dx
    y_axis = torch.arange(height + 1, device=device, dtype=dtype) * dy
    return _meshgrid(x_axis, y_axis)


def cell_center_coordinates(
    height: int, width: int, dx: float, dy: float, device=None, dtype=torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Physical ``(x, y)`` coordinates of the cell centers, each shape ``(H, W)``,
    at ``x = (i + 0.5) * dx`` and ``y = (j + 0.5) * dy``."""
    x_axis = (torch.arange(width, device=device, dtype=dtype) + 0.5) * dx
    y_axis = (torch.arange(height, device=device, dtype=dtype) + 0.5) * dy
    return _meshgrid(x_axis, y_axis)
