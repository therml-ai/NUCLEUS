from functools import lru_cache

import numpy as np
import torch

# NOTE: these utilities hardcode boundaries for pool-boiling.
# I.e., left, right, and bottom are no-slip walls. The top is an outflow.

# Physical grid spacing (dx = dy) of the pool-boiling simulations: 32 cells per unit
# length. Single source of truth shared by the dataset, model, and normalizer so the
# curl/grad reconstruction and the potential normalization stay consistent.
GRID_SPACING = 1 / 32

def divergence_centers_from_faces(facex, facey, dx, dy):
    """cell-centered divergence of a face-valued field.
    """
    return_numpy = isinstance(facex, np.ndarray)
    if return_numpy:
        facex = torch.from_numpy(facex)
        facey = torch.from_numpy(facey)
    div = torch.diff(facex, axis=-1) / dx + torch.diff(facey, axis=-2) / dy
    if return_numpy:
        return div.numpy()
    return div


def vorticity_nodes_from_faces(facex, facey, dx, dy):
    """Vorticity ``d vely/dx - d velx/dy`` at the interior nodes solved for
    (j=1..H, i=1..W-1), shape ``(..., H, W-1)``. The top row uses the
    Neumann-outflow ghost ``facex[H] = -facex[H-1]`` (odd reflection: velx =
    d psi/dy = 0 at the outflow). Differences are taken on the trailing axes so
    leading (batch/time) dims broadcast."""
    fx = torch.cat([facex, -facex[..., -1:, :]], dim=-2)         # (..., H+1, W+1)
    dvely_dx = torch.diff(facey, axis=-1) / dx                    # (..., H+1, W-1)
    dvelx_dy = torch.diff(fx, axis=-2) / dy                       # (..., H, W+1)
    return dvely_dx[..., 1:, :] - dvelx_dy[..., :, 1:-1]          # (..., H, W-1)


def grad_faces_from_centers(center, dx, dy):
    """Face-valued gradient of a cell-centered field, matching the potential's
    BCs: Neumann (zero) on the closed left/right/bottom walls and Dirichlet
    (phi = 0) at the top outflow (a half cell above the top row of centers)."""
    grad_x = torch.zeros(center.shape[:-1] + (center.shape[-1] + 1,), device=center.device, dtype=center.dtype)
    grad_x[..., 1:-1] = torch.diff(center, axis=-1) / dx            # interior x-faces

    grad_y = torch.zeros(center.shape[:-2] + (center.shape[-2] + 1, center.shape[-1]), device=center.device, dtype=center.dtype)
    grad_y[..., 1:-1, :] = torch.diff(center, axis=-2) / dy         # interior y-faces
    grad_y[..., -1, :] = -center[..., -1, :] / (dy / 2)          # Dirichlet top outflow
    return grad_x, grad_y


def curl_faces_from_nodes(center, dx, dy):
    """Face-valued curl of a nodal streamfunction: ``velfacex = d psi/dy`` on
    x-faces, ``velfacey = -d psi/dx`` on y-faces."""
    facex = torch.diff(center, axis=-2) / dy       # (..., H, W+1)
    facey = -torch.diff(center, axis=-1) / dx      # (..., H+1, W)
    return facex, facey


def _dct2_matrix(n: int, device, dtype) -> torch.Tensor:
    # Orthonormal DCT-II basis: C[k, i] = a(k) cos(pi (2i+1) k / (2n)), with
    # a(0) = sqrt(1/n) and a(k) = sqrt(2/n) otherwise. Its transpose is the inverse.
    freqs = torch.arange(n, device=device, dtype=dtype)
    samples = torch.arange(n, device=device, dtype=dtype)
    basis = torch.cos(torch.pi * (2 * samples[None, :] + 1) * freqs[:, None] / (2 * n))
    # DC (k=0) column is scaled by sqrt(1/n), the rest by sqrt(2/n).
    scale = torch.where(
        freqs == 0,
        torch.tensor((1.0 / n) ** 0.5, device=device, dtype=dtype),
        torch.tensor((2.0 / n) ** 0.5, device=device, dtype=dtype),
    )[:, None]
    return basis * scale


def _dct4_matrix(n: int, device, dtype) -> torch.Tensor:
    # Orthonormal DCT-IV basis: C[k, i] = sqrt(2/n) cos(pi (2i+1)(2k+1) / (4n)).
    # DCT-IV is its own inverse basis (the matrix is symmetric-orthonormal).
    freqs = torch.arange(n, device=device, dtype=dtype)
    samples = torch.arange(n, device=device, dtype=dtype)
    basis = torch.cos(torch.pi * (2 * samples[None, :] + 1) * (2 * freqs[:, None] + 1) / (4 * n))
    return basis * (2.0 / n) ** 0.5


def _dst2_matrix(n: int, device, dtype) -> torch.Tensor:
    # Orthonormal DST-II basis: S[k, i] = a(k) sin(pi (2i+1)(k+1) / (2n)), with
    # a(k) = sqrt(2/n) except the Nyquist row (k = n-1) which is sqrt(1/n). Its
    # transpose is the inverse.
    freqs = torch.arange(n, device=device, dtype=dtype)
    samples = torch.arange(n, device=device, dtype=dtype)
    basis = torch.sin(torch.pi * (2 * samples[None, :] + 1) * (freqs[:, None] + 1) / (2 * n))
    scale = torch.where(
        freqs == n - 1,
        torch.tensor((1.0 / n) ** 0.5, device=device, dtype=dtype),
        torch.tensor((2.0 / n) ** 0.5, device=device, dtype=dtype),
    )[:, None]
    return basis * scale


def _dst4_matrix(n: int, device, dtype) -> torch.Tensor:
    # Orthonormal DST-IV basis: S[k, i] = sqrt(2/n) sin(pi (2i+1)(2k+1) / (4n)).
    freqs = torch.arange(n, device=device, dtype=dtype)
    samples = torch.arange(n, device=device, dtype=dtype)
    basis = torch.sin(torch.pi * (2 * samples[None, :] + 1) * (2 * freqs[:, None] + 1) / (4 * n))
    return basis * (2.0 / n) ** 0.5


def solve_poisson_dirichlet_neumann(source: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Solve ``laplacian(psi) = source`` on a cell-centered grid with Dirichlet
    (``psi = 0``) on the closed left/right/bottom walls and Neumann
    (``d psi/dy = 0``) at the top outflow -- the streamfunction's boundary
    conditions (each closed wall is a streamline). The eigenbasis is DST-II in x
    (Dirichlet both walls) and DST-IV in y (Dirichlet bottom, Neumann top).
    ``source`` has shape ``(..., H, W)``; leading dims broadcast. Differentiable.
    """
    height, width = source.shape[-2], source.shape[-1]
    device, dtype = source.device, source.dtype

    basis_x = _dst2_matrix(width, device, dtype)     # (W, W)
    basis_y = _dst4_matrix(height, device, dtype)    # (H, H)

    freq_x = torch.arange(width, device=device, dtype=dtype)
    freq_y = torch.arange(height, device=device, dtype=dtype)
    eig_x = -4.0 / dx**2 * torch.sin((freq_x + 1) * torch.pi / (2 * width)) ** 2
    eig_y = -4.0 / dy**2 * torch.sin((2 * freq_y + 1) * torch.pi / (4 * height)) ** 2
    eigenvalues = eig_y[:, None] + eig_x[None, :]    # strictly negative -> invertible

    source_hat = torch.matmul(basis_y, torch.matmul(source, basis_x.transpose(-1, -2)))
    psi_hat = source_hat / eigenvalues
    return torch.matmul(basis_y.transpose(-1, -2), torch.matmul(psi_hat, basis_x))


def solve_poisson_neumann_dirichlet(source: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Solve ``laplacian(phi) = source`` on a cell-centered grid.

    NOTE: This is hard-coded for a pool-boiling setup.
    Boundary conditions: Neumann (``d phi/dn = 0``) on the left/right/bottom walls
    and Dirichlet (``phi = 0``) at the top outflow, matching
    
    Args:
        source: (..., H, W), 
        dx: float
        dy: float
    """
    height, width = source.shape[-2], source.shape[-1]
    device, dtype = source.device, source.dtype

    basis_x = _dct2_matrix(width, device, dtype)     # (W, W)
    basis_y = _dct4_matrix(height, device, dtype)    # (H, H)

    freq_x = torch.arange(width, device=device, dtype=dtype)
    freq_y = torch.arange(height, device=device, dtype=dtype)
    eig_x = -4.0 / dx**2 * torch.sin(freq_x * torch.pi / (2 * width)) ** 2
    eig_y = -4.0 / dy**2 * torch.sin((2 * freq_y + 1) * torch.pi / (4 * height)) ** 2
    eigenvalues = eig_y[:, None] + eig_x[None, :]    # strictly negative -> invertible

    # forward transform: source_hat = C4_y @ source @ C2_x^T
    source_hat = torch.matmul(basis_y, torch.matmul(source, basis_x.transpose(-1, -2)))
    phi_hat = source_hat / eigenvalues
    # inverse transform: phi = C4_y^T @ phi_hat @ C2_x
    return torch.matmul(basis_y.transpose(-1, -2), torch.matmul(phi_hat, basis_x))


@lru_cache(maxsize=None)
def _nodal_laplacian_eigsystem(height, width, dx, dy, device, dtype):
    """Eigen-decomposition of the separable nodal Laplacian for the streamfunction
    solve, cached per (height, width, dx, dy, device, dtype).

    x acts on interior nodes i=1..width-1 with Dirichlet walls (symmetric
    tridiagonal). y acts on nodes j=1..height with Dirichlet at the bottom wall
    and Neumann at the top outflow -- the ghost ``psi[H+1] = psi[H-1]`` makes the
    top row non-symmetric, so it is diagonalized directly (``torch.linalg.eig``).
    Only the source is differentiated through; the eigenvectors are grid constants.
    """
    main_x = -2.0 * torch.ones(width - 1, device=device, dtype=dtype)
    off_x = torch.ones(width - 2, device=device, dtype=dtype)
    laplacian_x = (torch.diag(main_x) + torch.diag(off_x, 1) + torch.diag(off_x, -1)) / dx**2

    main_y = -2.0 * torch.ones(height, device=device, dtype=dtype)
    off_y = torch.ones(height - 1, device=device, dtype=dtype)
    laplacian_y = torch.diag(main_y) + torch.diag(off_y, 1) + torch.diag(off_y, -1)
    laplacian_y[-1, -2] = 2.0  # Neumann top ghost
    laplacian_y = laplacian_y / dy**2

    eig_x, vec_x = torch.linalg.eigh(laplacian_x) # symmetric
    eig_y, vec_y = torch.linalg.eig(laplacian_y)  # non-symmetric top row
    eig_y, vec_y = eig_y.real, vec_y.real
    return eig_x, vec_x, vec_x.transpose(-1, -2), eig_y, vec_y, torch.linalg.inv(vec_y)


def stream_function_from_faces(facex, facey, dx, dy):
    """Nodal streamfunction psi ``(..., H+1, W+1)`` of a face-valued velocity
    field: solves ``laplacian(psi) = -vorticity`` with Dirichlet walls (psi = 0,
    each closed wall a streamline) and Neumann at the top outflow. The Dirichlet
    walls are left zeroed in the returned array."""
    height = facey.shape[-2] - 1
    width = facex.shape[-1] - 1
    omega = vorticity_nodes_from_faces(facex, facey, dx, dy)   # (..., H, W-1)

    eig_x, vec_x, vec_x_inv, eig_y, vec_y, vec_y_inv = _nodal_laplacian_eigsystem(
        height, width, dx, dy, omega.device, omega.dtype
    )
    rhs_hat = torch.matmul(vec_y_inv, torch.matmul(-omega, vec_x_inv.transpose(-1, -2)))
    psi_hat = rhs_hat / (eig_y[:, None] + eig_x[None, :])
    psi_interior = torch.matmul(vec_y, torch.matmul(psi_hat, vec_x.transpose(-1, -2)))

    psi = torch.zeros(omega.shape[:-2] + (height + 1, width + 1), device=omega.device, dtype=omega.dtype)
    psi[..., 1:, 1:width] = psi_interior # Dirichlet walls stay 0
    return psi


def helmholtz_from_faces(facex, facey, dx, dy):
    """Staggered Helmholtz decomposition of a face-valued velocity field into the
    nodal streamfunction psi ``(..., H+1, W+1)`` (solenoidal part) and the
    cell-centered potential phi ``(..., H, W)`` (dilatational part).

    Accepts either torch tensors or numpy arrays. If given numpy arrays the
    outputs are returned as numpy arrays, so numpy callers never see a tensor."""
    return_numpy = isinstance(facex, np.ndarray)
    if return_numpy:
        facex = torch.from_numpy(facex)
        facey = torch.from_numpy(facey)

    div = divergence_centers_from_faces(facex, facey, dx, dy)
    phi_centers = solve_poisson_neumann_dirichlet(div, dx, dy)
    psi_nodes = stream_function_from_faces(facex, facey, dx, dy)

    if return_numpy:
        return psi_nodes.numpy(), phi_centers.numpy()
    return psi_nodes, phi_centers


def reconstruct_velocity_from_helmholtz(psi_nodes, phi_centers, dx, dy):
    """Reconstruct the face velocity ``curl(psi) + grad(phi)`` from the nodal
    streamfunction and cell-centered potential."""
    curl_x, curl_y = curl_faces_from_nodes(psi_nodes, dx, dy)
    grad_x, grad_y = grad_faces_from_centers(phi_centers, dx, dy)
    return curl_x + grad_x, curl_y + grad_y