import torch
from torch.utils.cpp_extension import load

_sdf_reinit_cpp = load(
    "sdf_reinit",
    sources=["csrc/sdf_reinit.cpp"],
    extra_cflags=["-O3"],
    extra_link_args=["-fopenmp"],
    verbose=True
)

def sdf_reinit_drift(
    sdf_before: torch.Tensor,
    sdf_after:  torch.Tensor,
    dx: float,
) -> float:
    """
    Measures how much the zero level set moved after redistancing. Should be much less than dx.
    """
    near_interface = torch.abs(sdf_before) < 3 * dx
    if not near_interface.any():
        return 0.0
    drift = torch.abs(sdf_after[near_interface] - sdf_before[near_interface])
    return drift.mean().item()

def verify_sdf(sdf, dx, dy=None):
    if dy is None: dy = dx
    # Verify it's a distance function. |grad(sdf)| mean should be ~1, std should be ~0.
    grad_y, grad_x = torch.gradient(sdf, spacing=(dy, dx), dim=(-2, -1), edge_order=1)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    return grad_magnitude.mean(dim=(-2, -1)), grad_magnitude.std(dim=(-2, -1))

def sdf_reinit_fast_marching(sdf_init, dx, scale_factor=8, far_threshold=4.0):
    assert sdf_init.device == torch.device("cpu"), "SDF must be on CPU for fast marching reinitialization"
    return _sdf_reinit_cpp.sdf_reinit(sdf_init, dx, scale_factor, far_threshold)

def sdf_reinit_sussman(
    sdf0: torch.Tensor,
    dx: float,
    dy: float = None,
    n_iter: int = 5,
    dtau: float = None
) -> torch.Tensor:
    """
    Convention: sdf > 0 = vapor, sdf < 0 = liquid.
    Args:
        sdf0:   (H, W) tensor
        dx:     grid spacing in x
        dy:     grid spacing in y (defaults to dx)
        n_iter: pseudo-time iterations (5 is enough for in-training use)
        dtau:   pseudo-timestep (defaults to 0.5 * dx)
    Returns:
        sdf:    redistanced SDF, same shape as sdf0
    """
    
    if dy   is None: dy = dx
    if dtau is None: dtau = 0.3 * min(dx, dy)
    eps = 1e-6

    dsdf_dx = _ddx(sdf0, dx)
    dsdf_dy = _ddy(sdf0, dy)
    grad_mag0 = torch.sqrt(dsdf_dx**2 + dsdf_dy**2 + eps)
    S = sdf0 / torch.sqrt(sdf0**2 + (grad_mag0 * dx)**2 + eps)
    S = S.detach() # smoothed sdf is frozen across iterations and not backpropped through

    sdf = sdf0
    for _ in range(n_iter):
        sdf_prev = sdf.clone()
        grad_mag = godunov_grad_mag(sdf, S, dx, dy, eps)
        sdf = sdf - dtau * S * (grad_mag - 1.0)
        # Early stopping if sdf is not changing much
        if torch.abs(sdf - sdf_prev).max() < 1e-6:
            break

    return sdf

def godunov_grad_mag(sdf, S, dx, dy, eps):
    # One-sided differences via convolution.
    Dxm, Dxp = _one_sided_x(sdf, dx)
    Dym, Dyp = _one_sided_y(sdf, dy)
    ax = torch.where(
        S > 0,
        torch.relu( Dxm)**2 + torch.relu(-Dxp)**2, # vapor
        torch.relu(-Dxm)**2 + torch.relu( Dxp)**2  # liquid
    )
    ay = torch.where(
        S > 0,
        torch.relu( Dym)**2 + torch.relu(-Dyp)**2, # vapor
        torch.relu(-Dym)**2 + torch.relu( Dyp)**2  # liquid
    )
    grad_mag = torch.sqrt(ax + ay + eps)
    return grad_mag

def _replicate_pad_h(sdf: torch.Tensor, pad_top: int = 0, pad_bottom: int = 0) -> torch.Tensor:
    """Replicate-pad height dimension for a (H, W) tensor."""
    parts = []
    if pad_top > 0:
        parts.append(sdf[:1, :].expand(pad_top, -1))
    parts.append(sdf)
    if pad_bottom > 0:
        parts.append(sdf[-1:, :].expand(pad_bottom, -1))
    return torch.cat(parts, dim=0)

def _replicate_pad_w(sdf: torch.Tensor, pad_left: int = 0, pad_right: int = 0) -> torch.Tensor:
    """Replicate-pad width dimension for a (H, W) tensor."""
    parts = []
    if pad_left > 0:
        parts.append(sdf[:, :1].expand(-1, pad_left))
    parts.append(sdf)
    if pad_right > 0:
        parts.append(sdf[:, -1:].expand(-1, pad_right))
    return torch.cat(parts, dim=1)

def _ddx(sdf: torch.Tensor, dx: float) -> torch.Tensor:
    """Central difference in x using replicate padding."""
    sdf_pad = _replicate_pad_h(sdf, pad_top=1, pad_bottom=1)
    return (sdf_pad[2:, :] - sdf_pad[:-2, :]) / (2 * dx)

def _ddy(sdf: torch.Tensor, dy: float) -> torch.Tensor:
    """Central difference in y using replicate padding."""
    sdf_pad = _replicate_pad_w(sdf, pad_left=1, pad_right=1)
    return (sdf_pad[:, 2:] - sdf_pad[:, :-2]) / (2 * dy)

def _one_sided_x(sdf: torch.Tensor, dx: float):
    """Forward and backward differences in x."""
    sdf_pad = _replicate_pad_h(sdf, pad_top=1, pad_bottom=1)
    Dxm = (sdf_pad[1:-1, :] - sdf_pad[:-2, :]) / dx   # backward
    Dxp = (sdf_pad[2:,   :] - sdf_pad[1:-1, :]) / dx  # forward
    return Dxm, Dxp

def _one_sided_y(sdf: torch.Tensor, dy: float):
    """Forward and backward differences in y."""
    sdf_pad = _replicate_pad_w(sdf, pad_left=1, pad_right=1)
    Dym = (sdf_pad[:, 1:-1] - sdf_pad[:, :-2]) / dy   # backward
    Dyp = (sdf_pad[:, 2:  ] - sdf_pad[:, 1:-1]) / dy  # forward
    return Dym, Dyp
