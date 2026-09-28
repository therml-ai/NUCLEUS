"""Leray-Hodge projection of a staggered velocity field onto its divergence-free part."""

from typing import Tuple

import torch

from nucleus.physics.poisson import (
    divergence_centers_from_faces,
    grad_faces_from_centers,
    solve_poisson_neumann_dirichlet,
)


def leray_projection(
    facex: torch.Tensor,
    facey: torch.Tensor,
    dx: float,
    dy: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project a MAC face velocity onto its divergence-free part: ``P(u) = u - grad(phi)``
    where ``laplacian(phi) = div(u)``.
    """
    divergence = divergence_centers_from_faces(facex, facey, dx, dy)
    # float64 keeps the DCT-based solve's residual well below float32 noise, so the
    # projected field is divergence-free to numerical precision rather than ~1e-2.
    phi = solve_poisson_neumann_dirichlet(divergence.to(torch.float64), dx, dy)
    grad_x, grad_y = grad_faces_from_centers(phi, dx, dy)
    return facex - grad_x.to(facex.dtype), facey - grad_y.to(facey.dtype)