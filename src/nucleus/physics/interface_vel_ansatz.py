"""Interface velocity jump enforced by the additive ansatz ``u = NN + S * H(sdf) * n``,
with ``S = mdot * (1/rho_v - 1/rho_l)``, so ``u_v - u_l = S * n`` holds by construction
and the network predicts the continuous remainder. The jump is purely dilatational; it
never belongs in a ``curl(psi)`` part."""

from typing import Tuple

import torch

from nucleus.physics.poisson import GRID_SPACING
from nucleus.physics.sdf import interface_normals


def smoothed_heaviside(sdf: torch.Tensor, epsilon: float) -> torch.Tensor:
    """``0.5 * (1 + tanh(sdf / epsilon))`` on ``(..., H, W)``."""
    return 0.5 * (1.0 + torch.tanh(sdf / epsilon))


def interface_jump_magnitude(
    mdot: torch.Tensor, rho_vapor: float, rho_liquid: float = 1.0
) -> torch.Tensor:
    """``S = mdot * (1/rho_vapor - 1/rho_liquid)``; ``mdot`` is a scalar or ``(..., H, W)``."""
    return mdot * (1.0 / rho_vapor - 1.0 / rho_liquid)


def _center_to_x_face(center: torch.Tensor) -> torch.Tensor:
    """``(..., H, W)`` -> ``(..., H, W + 1)`` by averaging; wall faces take the
    adjacent center."""
    interior = 0.5 * (center[..., :, :-1] + center[..., :, 1:])
    return torch.cat([center[..., :, :1], interior, center[..., :, -1:]], dim=-1)


def _center_to_y_face(center: torch.Tensor) -> torch.Tensor:
    """``(..., H, W)`` -> ``(..., H + 1, W)`` by averaging; wall faces take the
    adjacent center."""
    interior = 0.5 * (center[..., :-1, :] + center[..., 1:, :])
    return torch.cat([center[..., :1, :], interior, center[..., -1:, :]], dim=-2)


def interface_jump_velocity(
    sdf: torch.Tensor,
    mdot: torch.Tensor,
    rho_vapor: float,
    rho_liquid: float = 1.0,
    dx: float = GRID_SPACING,
    dy: float = GRID_SPACING,
    epsilon: float = 2.0 * GRID_SPACING,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The jump field ``S * H(sdf) * n`` on the MAC faces: ``(jump_facex, jump_facey)``
    of shapes ``(..., H, W + 1)`` and ``(..., H + 1, W)`` from ``sdf`` ``(..., H, W)``.
    ``epsilon`` is the Heaviside smoothing width."""
    normal_x, normal_y = interface_normals(sdf, dx, dy)
    jump = interface_jump_magnitude(mdot, rho_vapor, rho_liquid) * smoothed_heaviside(sdf, epsilon)
    return _center_to_x_face(jump * normal_x), _center_to_y_face(jump * normal_y)


def interface_vel_ansatz(
    model_velx: torch.Tensor,
    model_vely: torch.Tensor,
    sdf: torch.Tensor,
    mdot: torch.Tensor,
    rho_vapor: float,
    rho_liquid: float = 1.0,
    dx: float = GRID_SPACING,
    dy: float = GRID_SPACING,
    epsilon: float = 2.0 * GRID_SPACING,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``u = NN + S * H(sdf) * n`` on a MAC velocity: ``model_velx`` ``(..., H, W + 1)``
    and ``model_vely`` ``(..., H + 1, W)`` in, ``(velx, vely)`` of the same shapes out.
    Arguments as in ``interface_jump_velocity``."""
    jump_facex, jump_facey = interface_jump_velocity(
        sdf, mdot, rho_vapor, rho_liquid, dx, dy, epsilon
    )
    return model_velx + jump_facex, model_vely + jump_facey
