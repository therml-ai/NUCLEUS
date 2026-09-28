from typing import Tuple

import torch

from nucleus.physics.sdf import vapor_mask, liquid_mask

# Ghost-fluid temperature gradient for each phase.
#
# The temperature is discontinuous in slope across the liquid-vapor interface,
# so a plain central difference near the interface mixes the two phases and
# smears the flux that drives phase change. The ghost fluid method fixes this:
# when the finite-difference stencil for one phase reaches across the interface
# into the other phase, the neighbor's temperature is replaced by a linear
# extrapolation of *this* phase's field up to the known interface (saturation)
# temperature. For a center cell i whose neighbor i+1 is in the other phase,
#
#     ghost(T)_{i+1} = T_i + (T_sat - T_i) / theta
#
# where theta in (0, 1] is the sub-cell distance from cell i to the interface as
# a fraction of the cell spacing, taken from the SDF: theta = |sdf_i| / (|sdf_i| +
# |sdf_{i+1}|) (for a true distance function this is just |sdf_i| / spacing, the
# normalized interface distance). The central difference then recovers this
# phase's one-sided gradient at the interface instead of a smeared one.


def _replicate_pad_hw(field: torch.Tensor) -> torch.Tensor:
    """Replicate-pad the trailing (H, W) dims by one cell on every side, for any
    leading dims (torch's F.pad replicate is limited to 3-5D)."""
    field = torch.cat([field[..., :1, :], field, field[..., -1:, :]], dim=-2)
    field = torch.cat([field[..., :, :1], field, field[..., :, -1:]], dim=-1)
    return field


def _pad_temp_hw(field: torch.Tensor, wall_temp) -> torch.Tensor:
    """Pad temperature by one cell on every side. The bottom row (index 0 on the
    ``H`` axis, the heater wall for ``origin='lower'`` data) uses a Dirichlet ghost
    reflecting about ``wall_temp`` -- ``2 * wall_temp - T[0]`` -- so a central
    difference sees the fixed wall temperature at the wall face (half a cell below
    the first cell center) instead of the zero-gradient a replicate pad imposes.
    All other edges replicate. When ``wall_temp is None`` the bottom also
    replicates (no wall BC applied)."""
    if wall_temp is None:
        bottom_ghost = field[..., :1, :]
    else:
        bottom_ghost = 2.0 * wall_temp - field[..., :1, :]
    field = torch.cat([bottom_ghost, field, field[..., -1:, :]], dim=-2)
    field = torch.cat([field[..., :, :1], field, field[..., :, -1:]], dim=-1)
    return field


def _ghost_fluid_gradient(
    temp: torch.Tensor,
    sdf: torch.Tensor,
    sat_temp,
    dx: float,
    dy: float,
    phase_mask: torch.Tensor,
    opposite_is_vapor: bool,
    eps: float,
    wall_temp=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    padded_temp = _pad_temp_hw(temp, wall_temp)
    padded_sdf = _replicate_pad_hw(sdf)
    center_temp = padded_temp[..., 1:-1, 1:-1]
    center_sdf = padded_sdf[..., 1:-1, 1:-1]

    def effective(neighbor_temp: torch.Tensor, neighbor_sdf: torch.Tensor) -> torch.Tensor:
        # A neighbor is "across the interface" when it is in the other phase.
        across = neighbor_sdf >= 0 if opposite_is_vapor else neighbor_sdf < 0
        # theta: sub-cell distance from the center cell to the interface, in cell
        # fractions. clamp keeps the ghost finite when the interface sits on the
        # center cell (sdf ~ 0).
        theta = (center_sdf.abs() / (center_sdf.abs() + neighbor_sdf.abs() + eps))#.clamp_min(eps)).clamp_min(eps)
        ghost = center_temp + (sat_temp - center_temp) / (theta + eps)
        return torch.where(across, ghost, neighbor_temp)

    right = effective(padded_temp[..., 1:-1, 2:], padded_sdf[..., 1:-1, 2:])
    left = effective(padded_temp[..., 1:-1, :-2], padded_sdf[..., 1:-1, :-2])
    up = effective(padded_temp[..., 2:, 1:-1], padded_sdf[..., 2:, 1:-1])
    down = effective(padded_temp[..., :-2, 1:-1], padded_sdf[..., :-2, 1:-1])

    grad_x = (right - left) / (2 * dx) * phase_mask
    grad_y = (up - down) / (2 * dy) * phase_mask
    return grad_x, grad_y


def vapor_temp_grad(
    temp: torch.Tensor, sdf: torch.Tensor, sat_temp, dx: float, dy: float,
    wall_temp=None, eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ghost-fluid temperature gradient of the vapor phase.

    Central differences of the temperature, but where a stencil reaches into the
    liquid the liquid neighbor is replaced by a vapor ghost extrapolated to the
    interface (saturation) temperature, so the gradient reflects only the vapor
    field. The result is masked to the vapor (``sdf >= 0``) and is zero in the
    liquid.

    Args:
        temp: cell-centered temperature, shape ``(..., H, W)``. x is the last axis
            (width), y the second-to-last (height).
        sdf: cell-centered signed distance, shape ``(..., H, W)``. sdf < 0 is
            liquid, sdf >= 0 is vapor.
        sat_temp: interface (saturation) temperature; a scalar or a tensor
            broadcastable to ``temp``.
        dx: cell spacing in x.
        dy: cell spacing in y.
        wall_temp: heater temperature at the bottom wall (index 0 on ``H``), in the
            same units as ``temp``. Applies a Dirichlet BC there instead of the
            default zero-gradient. ``None`` keeps the zero-gradient wall.
        eps: floor on the sub-cell interface distance, guarding the ghost against
            division by zero when the interface lies on a cell.

    Returns:
        ``(grad_x, grad_y)``, each shape ``(..., H, W)``.
    """
    return _ghost_fluid_gradient(
        temp, sdf, sat_temp, dx, dy, vapor_mask(sdf), opposite_is_vapor=False,
        eps=eps, wall_temp=wall_temp,
    )


def liquid_temp_grad(
    temp: torch.Tensor, sdf: torch.Tensor, sat_temp, dx: float, dy: float,
    wall_temp=None, eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ghost-fluid temperature gradient of the liquid phase.

    The mirror of ``vapor_temp_grad``: where a stencil reaches into the vapor the
    vapor neighbor is replaced by a liquid ghost extrapolated to the interface
    (saturation) temperature. The result is masked to the liquid (``sdf < 0``) and
    is zero in the vapor.

    Args:
        temp: cell-centered temperature, shape ``(..., H, W)``. x is the last axis
            (width), y the second-to-last (height).
        sdf: cell-centered signed distance, shape ``(..., H, W)``. sdf < 0 is
            liquid, sdf >= 0 is vapor.
        sat_temp: interface (saturation) temperature; a scalar or a tensor
            broadcastable to ``temp``.
        dx: cell spacing in x.
        dy: cell spacing in y.
        wall_temp: heater temperature at the bottom wall (index 0 on ``H``), in the
            same units as ``temp``. Applies a Dirichlet BC there instead of the
            default zero-gradient. ``None`` keeps the zero-gradient wall.
        eps: floor on the sub-cell interface distance, guarding the ghost against
            division by zero when the interface lies on a cell.

    Returns:
        ``(grad_x, grad_y)``, each shape ``(..., H, W)``.
    """
    return _ghost_fluid_gradient(
        temp, sdf, sat_temp, dx, dy, liquid_mask(sdf), opposite_is_vapor=True,
        eps=eps, wall_temp=wall_temp,
    )
