import math
from typing import Tuple

import torch


def sign_heaviside(phi: torch.Tensor) -> torch.Tensor:
    """``(sign(1, phi) + 1) / 2`` on ``(..., H, W)``; ``phi == 0`` counts as vapor, as
    in Fortran's ``SIGN``."""
    return (phi >= 0).to(phi.dtype)


def smoothed_heaviside(phi: torch.Tensor, smear) -> torch.Tensor:
    """Heaviside of ``phi`` ``(..., H, W)`` with a sine ramp over ``|phi| <= smear``,
    from ``Stencils_lsCenterPropsSmeared``."""
    ramp = 0.5 + phi / (2.0 * smear) + torch.sin(math.pi * phi / smear) / (2.0 * math.pi)
    return torch.where(phi.abs() <= smear, ramp, sign_heaviside(phi))


def smooth_density(
    phi: torch.Tensor, rho_gas, rho_liquid, smear
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(rhoc, smhv)`` from ``Multiphase_setFluidProps``, each ``(..., H, W)``:
    ``smhv * rho_gas + (1 - smhv) * rho_liquid``.

    Flash-X's RHOC_VAR is the specific volume ``1/rho``, so the equivalent call is
    ``rho_gas=1/rhogas, rho_liquid=1``. That matters: ``level_set_normals``
    differentiates this field, and only the specific volume increases into the
    vapor so the normals point liquid -> vapor like ``grad(phi)``."""
    smhv = smoothed_heaviside(phi, smear)
    rhoc = smhv * rho_gas + (1.0 - smhv) * rho_liquid
    return rhoc, smhv


def min_cell_diag(dx: float, dy: float) -> float:
    """Cell diagonal; Flash-X smears properties over ``mph_iPropSmear`` of these."""
    return math.sqrt(dx * dx + dy * dy)


def level_set_normals(
    phi: torch.Tensor, dx: float, dy: float, tol: float = 1e-13
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalized central-difference gradient of ``phi`` ``(..., H, W)``, from
    ``Stencils_lsNormals2d``. Returns ``(normal_x, normal_y)``, each ``(..., H, W)``
    and zero on the 1-cell border. Flash-X calls this on the smoothed ``rhoc``."""
    grad_x = (phi[..., 1:-1, 2:] - phi[..., 1:-1, :-2]) / (2.0 * dx)
    grad_y = (phi[..., 2:, 1:-1] - phi[..., :-2, 1:-1]) / (2.0 * dy)
    magnitude = torch.sqrt(grad_x * grad_x + grad_y * grad_y + tol)
    normal_x = torch.nn.functional.pad(grad_x / magnitude, (1, 1, 1, 1))
    normal_y = torch.nn.functional.pad(grad_y / magnitude, (1, 1, 1, 1))
    return normal_x, normal_y
