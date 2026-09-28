"""Projection of a staggered velocity field onto a prescribed divergence."""

from typing import Optional, Tuple

import torch

from nucleus.physics.poisson import (
    divergence_centers_from_faces,
    grad_faces_from_centers,
    solve_poisson_neumann_dirichlet,
)


def div_projection(
    facex: torch.Tensor,
    facey: torch.Tensor,
    dx: float,
    dy: float,
    source: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project a MAC face velocity so its divergence matches ``source``:
    ``P(u) = u - grad(phi)`` where ``laplacian(phi) = div(u) - source``.

    ``source`` is a cell-centered ``(..., H, W)`` tensor (e.g. the volumetric
    expansion from phase change); ``None`` gives the divergence-free projection.
    """
    divergence = divergence_centers_from_faces(facex, facey, dx, dy)
    if source is not None:
        divergence = divergence - source
    # float64 keeps the DCT-based solve's residual well below float32 noise, so the
    # projected field is divergence-free to numerical precision rather than ~1e-2.
    phi = solve_poisson_neumann_dirichlet(divergence.to(torch.float64), dx, dy)
    grad_x, grad_y = grad_faces_from_centers(phi, dx, dy)
    return facex - grad_x.to(facex.dtype), facey - grad_y.to(facey.dtype)
