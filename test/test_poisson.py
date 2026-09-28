import math
import numpy as np
import torch

from nucleus.physics.poisson import (
    helmholtz_from_faces,
    reconstruct_velocity_from_helmholtz,
    curl_faces_from_nodes,
    grad_faces_from_centers,
    divergence_centers_from_faces,
    solve_poisson_neumann_dirichlet,
    solve_poisson_dirichlet_neumann,
)


def _cell_centers(n, dx, dy):
    center_x = (np.arange(n) + 0.5) * dx
    center_y = (np.arange(n) + 0.5) * dy
    grid_y, grid_x = np.meshgrid(center_y, center_x, indexing="ij")  # (H, W)
    return grid_x, grid_y


def test_solver_recovers_a_known_potential():
    # Build a source as the discrete divergence of grad(phi_true) (Neumann walls,
    # Dirichlet top) and check the torch solver recovers that potential to machine
    # precision.
    n = 48
    dx = dy = 1.0 / n
    center_x, center_y = _cell_centers(n, dx, dy)
    length = n * dx

    phi_true = torch.from_numpy(
        np.cos(np.pi * center_x / length) * np.cos(0.5 * np.pi * center_y / length)
    )
    grad_x, grad_y = grad_faces_from_centers(phi_true, dx, dy)
    source = divergence_centers_from_faces(grad_x, grad_y, dx, dy)

    phi_torch = solve_poisson_neumann_dirichlet(source, dx, dy)

    assert torch.linalg.norm(phi_torch - phi_true) / torch.linalg.norm(phi_true) < 1e-6


def test_solver_reproduces_the_source_as_its_divergence():
    # laplacian(solve(source)) == source: the Poisson solve inverts the 5-point
    # Laplacian, so applying it back returns the source (interior/eigen-exact).
    n = 40
    dx = dy = 1.0 / n
    torch.manual_seed(0)
    source = torch.randn(2, n, n, dtype=torch.float64)

    phi = solve_poisson_neumann_dirichlet(source, dx, dy)

    # 5-point Laplacian with the solver's BCs: Neumann walls (edge value repeats)
    # and Dirichlet phi=0 a half cell above the top row.
    padded = torch.nn.functional.pad(phi, (1, 1, 1, 0), mode="replicate")   # Neumann L/R/bottom
    top_ghost = -phi[..., -1:, :]                                           # Dirichlet top (odd)
    padded = torch.cat([padded, torch.nn.functional.pad(top_ghost, (1, 1))], dim=-2)
    laplacian = (
        (padded[..., 1:-1, 2:] - 2 * phi + padded[..., 1:-1, :-2]) / dx**2
        + (padded[..., 2:, 1:-1] - 2 * phi + padded[..., :-2, 1:-1]) / dy**2
    )
    assert torch.linalg.norm(laplacian - source) / torch.linalg.norm(source) < 1e-6


def test_streamfunction_solver_reproduces_its_source():
    # laplacian(solve(source)) == source for the Dirichlet-wall / Neumann-top
    # streamfunction operator. Dirichlet walls -> odd ghost (psi = 0 half a cell
    # outside the first center); Neumann top -> even ghost.
    n = 40
    dx = dy = 1.0 / n
    torch.manual_seed(0)
    source = torch.randn(2, n, n, dtype=torch.float64)

    psi = solve_poisson_dirichlet_neumann(source, dx, dy)

    left = -psi[..., :, :1]                      # Dirichlet left wall (odd)
    right = -psi[..., :, -1:]                     # Dirichlet right wall (odd)
    padded = torch.cat([left, psi, right], dim=-1)
    bottom = -psi[..., :1, :]                     # Dirichlet bottom wall (odd)
    top = psi[..., -1:, :]                         # Neumann top (even)
    padded = torch.cat(
        [torch.nn.functional.pad(bottom, (1, 1)), padded, torch.nn.functional.pad(top, (1, 1))],
        dim=-2,
    )
    laplacian = (
        (padded[..., 1:-1, 2:] - 2 * psi + padded[..., 1:-1, :-2]) / dx**2
        + (padded[..., 2:, 1:-1] - 2 * psi + padded[..., :-2, 1:-1]) / dy**2
    )
    assert torch.linalg.norm(laplacian - source) / torch.linalg.norm(source) < 1e-6


def test_solver_is_differentiable_and_batches():
    n = 24
    dx = dy = 1.0 / n
    source = torch.randn(3, 5, n, n, dtype=torch.float64, requires_grad=True)

    phi = solve_poisson_neumann_dirichlet(source, dx, dy)
    assert phi.shape == source.shape

    phi.sum().backward()
    assert torch.isfinite(source.grad).all()


def test_helmholtz():
    # Round-trip: decompose a face velocity that satisfies the pool-boiling wall
    # BCs (no-penetration: velx=0 at the left/right walls, vely=0 at the bottom)
    # and check curl(psi) + grad(phi) reconstructs it. The field is built from a
    # nodal streamfunction that vanishes on the walls and a cell potential with
    # Neumann walls / Dirichlet top, so it lies in the decomposition's range. (A
    # wall-penetrating field, e.g. velx ~ cos(pi x) that is nonzero at x=0,1,
    # would not round-trip -- the decomposition forces velx=0 at closed walls.)
    H, W = 48, 40
    dx = dy = 1.0 / 32
    length_x, length_y = W * dx, H * dy

    node_x = (torch.arange(W + 1, dtype=torch.float64) * dx)[None, :]
    node_y = (torch.arange(H + 1, dtype=torch.float64) * dy)[:, None]
    psi_true = torch.sin(math.pi * node_x / length_x) * torch.sin(0.5 * math.pi * node_y / length_y)

    center_x = ((torch.arange(W, dtype=torch.float64) + 0.5) * dx)[None, :]
    center_y = ((torch.arange(H, dtype=torch.float64) + 0.5) * dy)[:, None]
    phi_true = torch.cos(math.pi * center_x / length_x) * torch.cos(0.5 * math.pi * center_y / length_y)

    curl_x, curl_y = curl_faces_from_nodes(psi_true, dx, dy)
    grad_x, grad_y = grad_faces_from_centers(phi_true, dx, dy)
    u, v = curl_x + grad_x, curl_y + grad_y

    psi, phi = helmholtz_from_faces(u, v, dx, dy)
    assert psi.shape == (H + 1, W + 1)   # nodal streamfunction
    assert phi.shape == (H, W)           # cell-centered potential

    rx, ry = reconstruct_velocity_from_helmholtz(psi, phi, dx, dy)
    assert torch.allclose(u, rx, atol=1e-6)
    assert torch.allclose(v, ry, atol=1e-6)