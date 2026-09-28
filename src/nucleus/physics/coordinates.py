"""Coordinate grids for the MAC layout: walls at the origin, cell centers at
``((i + 0.5) * dx, (j + 0.5) * dy)``, faces on the cell edges."""

from typing import Tuple

import torch


def domain_extent(height: int, width: int, dx: float, dy: float) -> Tuple[float, float]:
    """``(domain_width, domain_height)`` of a ``height`` x ``width`` cell grid."""
    return width * dx, height * dy


def _meshgrid(x_axis: torch.Tensor, y_axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(x_coords, y_coords)``, each ``(len(y_axis), len(x_axis))``."""
    y_coords, x_coords = torch.meshgrid(y_axis, x_axis, indexing="ij")
    return x_coords, y_coords


def x_face_coordinates(
    height: int, width: int, dx: float, dy: float, device=None, dtype=torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(x, y)`` of the x-velocity faces, each ``(H, W + 1)``; the outer faces sit
    on the left and right walls."""
    x_axis = torch.arange(width + 1, device=device, dtype=dtype) * dx
    y_axis = (torch.arange(height, device=device, dtype=dtype) + 0.5) * dy
    return _meshgrid(x_axis, y_axis)


def y_face_coordinates(
    height: int, width: int, dx: float, dy: float, device=None, dtype=torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(x, y)`` of the y-velocity faces, each ``(H + 1, W)``; the outer faces sit
    on the bottom wall and top outflow."""
    x_axis = (torch.arange(width, device=device, dtype=dtype) + 0.5) * dx
    y_axis = torch.arange(height + 1, device=device, dtype=dtype) * dy
    return _meshgrid(x_axis, y_axis)


def cell_center_coordinates(
    height: int, width: int, dx: float, dy: float, device=None, dtype=torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(x, y)`` of the cell centers, each ``(H, W)``."""
    x_axis = (torch.arange(width, device=device, dtype=dtype) + 0.5) * dx
    y_axis = (torch.arange(height, device=device, dtype=dtype) + 0.5) * dy
    return _meshgrid(x_axis, y_axis)
