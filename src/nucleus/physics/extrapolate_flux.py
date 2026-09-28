import torch

from nucleus.physics.sdf import vapor_mask, liquid_mask

def _upwind_normal_gradient(
    field: torch.Tensor, normal_x: torch.Tensor, normal_y: torch.Tensor, dx: float, dy: float
) -> torch.Tensor:
    """``n . grad(field)`` with first-order upwinding against the normal, so the
    stencil leans toward where the extrapolated information comes from (the
    interface). Shape ``(..., H, W)``."""
    forward_x = torch.zeros_like(field)
    forward_x[..., :, :-1] = (field[..., :, 1:] - field[..., :, :-1]) / dx
    backward_x = torch.zeros_like(field)
    backward_x[..., :, 1:] = (field[..., :, 1:] - field[..., :, :-1]) / dx

    forward_y = torch.zeros_like(field)
    forward_y[..., :-1, :] = (field[..., 1:, :] - field[..., :-1, :]) / dy
    backward_y = torch.zeros_like(field)
    backward_y[..., 1:, :] = (field[..., 1:, :] - field[..., :-1, :]) / dy

    grad_x = torch.where(normal_x > 0, backward_x, forward_x)
    grad_y = torch.where(normal_y > 0, backward_y, forward_y)
    return normal_x * grad_x + normal_y * grad_y


def extrapolate_phase_flux(
    q_l: torch.Tensor,
    q_v: torch.Tensor,
    sdf,
    normal_x: torch.Tensor,
    normal_y: torch.Tensor,
    dx: float,
    dy: float,
    tolerance: float = 1e-6,
    max_iterations: int = 5,
) -> torch.Tensor:
    r""" Aslam constant extrapolation of the phase heat fluxes across the interface.
    """
    time_step = 0.5 * min(dx, dy)
    ext_q_l = q_l.clone()
    ext_q_v = q_v.clone()

    vmask = vapor_mask(sdf).to(q_l.dtype)
    lmask = liquid_mask(sdf).to(q_v.dtype)

    for _ in range(max_iterations):
        ext_q_l -= time_step * vmask * _upwind_normal_gradient(ext_q_l, normal_x, normal_y, dx, dy)
        ext_q_v -= time_step * lmask * _upwind_normal_gradient(ext_q_v, -normal_x, -normal_y, dx, dy)        
    return ext_q_l, ext_q_v
