"""Hard enforcement of the liquid-vapor interface velocity jump via an additive
ansatz ``u = NN + S * H(sdf) * n``.

At a phase-change interface the fluid velocity jumps in the normal direction,
``u_v - u_l = mdot * (1/rho_v - 1/rho_l) * n``, where ``mdot`` is the mass flux, ``n``
the interface normal, and ``u_v``/``u_l`` the one-sided limits from the vapor/liquid
sides. This module builds an analytic field that carries exactly that jump and adds it
to the (continuous) network velocity, so the condition holds by construction rather
than being learned: the network predicts the continuous remainder ``NN = u - jump``.

The jump magnitude ``S = mdot * (1/rho_v - 1/rho_l)`` is multiplied by a smoothed
Heaviside of the SDF (0 in the liquid ``sdf < 0``, 1 in the vapor ``sdf >= 0``), so the
added field steps from 0 to ``S * n`` across the interface. On a MAC grid the step is
smeared over ``epsilon`` (a few cells); a true discontinuity is not representable.

The jump is purely normal and divergent (it is the volumetric expansion of
evaporation), so it belongs in the dilatational velocity channel, never in a
divergence-free ``curl(psi)`` part. The interface normal ``n = grad(sdf)/|grad(sdf)|``
points from liquid into vapor, matching the sign of the jump condition above.

Velocity lives on a staggered MAC grid: x-velocity on the vertical faces, shape
``(..., H, W + 1)``, and y-velocity on the horizontal faces, shape ``(..., H + 1, W)``.
The SDF, normals, and jump are cell-centered ``(..., H, W)`` and interpolated to the
faces.
"""

from typing import Tuple

import torch

from nucleus.physics.poisson import GRID_SPACING
from nucleus.physics.sdf import interface_normals


def smoothed_heaviside(sdf: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Smoothed Heaviside of the SDF: ~0 in the liquid (``sdf < 0``), ~1 in the vapor
    (``sdf > 0``), and 0.5 at the interface, transitioning over ~``epsilon``. Same shape
    as ``sdf``."""
    return 0.5 * (1.0 + torch.tanh(sdf / epsilon))


def interface_jump_magnitude(
    mdot: torch.Tensor, rho_vapor: float, rho_liquid: float = 1.0
) -> torch.Tensor:
    """Normal-velocity jump magnitude ``S = mdot * (1/rho_vapor - 1/rho_liquid)``. Same
    shape as ``mdot`` (which may be a scalar or a ``(..., H, W)`` field)."""
    return mdot * (1.0 / rho_vapor - 1.0 / rho_liquid)


def _center_to_x_face(center: torch.Tensor) -> torch.Tensor:
    """Average a cell-centered field ``(..., H, W)`` to the x-faces ``(..., H, W + 1)``.
    Interior faces average their two neighboring centers; the left/right wall faces take
    the adjacent center (one-sided)."""
    interior = 0.5 * (center[..., :, :-1] + center[..., :, 1:])
    return torch.cat([center[..., :, :1], interior, center[..., :, -1:]], dim=-1)


def _center_to_y_face(center: torch.Tensor) -> torch.Tensor:
    """Average a cell-centered field ``(..., H, W)`` to the y-faces ``(..., H + 1, W)``.
    Interior faces average their two neighboring centers; the bottom/top faces take the
    adjacent center (one-sided)."""
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
    """The analytic jump field ``S * H(sdf) * n``, interpolated to the MAC faces.

    Returns ``(jump_facex, jump_facey)`` with shapes ``(..., H, W + 1)`` and
    ``(..., H + 1, W)``. Adding these to a continuous velocity produces the jump
    ``u_v - u_l = mdot * (1/rho_v - 1/rho_l) * n`` across the interface.

    Args:
        sdf: cell-centered signed distance, shape ``(..., H, W)`` (liquid ``< 0``).
        mdot: mass flux, scalar or field broadcastable to ``(..., H, W)``.
        rho_vapor: vapor density.
        rho_liquid: liquid density.
        dx: cell spacing in x.
        dy: cell spacing in y.
        epsilon: smoothing width of the Heaviside step.
    """
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
    """Apply the interface jump ansatz ``u = NN + S * H(sdf) * n`` to a MAC velocity.

    ``model_velx``/``model_vely`` are the continuous network velocity on the x-faces
    ``(..., H, W + 1)`` and y-faces ``(..., H + 1, W)``. Returns the constrained
    ``(velx, vely)`` with the same shapes, whose one-sided limits satisfy
    ``u_v - u_l = mdot * (1/rho_v - 1/rho_l) * n`` across the interface regardless of the
    network. See :func:`interface_jump_velocity` for the arguments.
    """
    jump_facex, jump_facey = interface_jump_velocity(
        sdf, mdot, rho_vapor, rho_liquid, dx, dy, epsilon
    )
    return model_velx + jump_facex, model_vely + jump_facey
