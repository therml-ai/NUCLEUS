import pytest
import torch
from nucleus.utils.sdf_reinit import (
    sdf_reinit_fast_marching, 
    sdf_reinit_sussman,
    verify_sdf,
    sdf_reinit_drift
)

def make_circle_sdf(n=128, extent=8.0, radius=0.7, center=(0.1, -0.05), negate=True):
    x = torch.linspace(-extent, extent, n)
    y = torch.linspace(-extent, extent, n)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    cx, cy = center
    phi = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - radius
    # Convert inside circle to positive, outside to negative.
    if negate:
        phi = -phi
    dx = float(x[1] - x[0])
    return phi, dx


@pytest.mark.parametrize("noise_std", [0.05, 0.10])
def test_fast_marching_reinitializes_noisy_circle(noise_std):
    torch.manual_seed(0)

    phi_true, dx = make_circle_sdf()

    phi_noisy = phi_true + noise_std * torch.randn_like(phi_true)
    phi_reinit = sdf_reinit_fast_marching(
        phi_noisy.cpu(),  # fast marching impl only supports CPU
        dx=dx,
        scale_factor=8,
        far_threshold=4.0,
    )
    
    # Check |grad(phi)| stats improved toward 1
    mean_before, std_before = verify_sdf(phi_noisy, dx)
    mean_after, std_after = verify_sdf(phi_reinit, dx)
    err_before = torch.abs(mean_before - 1.0).item()
    err_after = torch.abs(mean_after - 1.0).item()
    assert err_after < err_before, f"mean |grad|-1 did not improve: {err_before} -> {err_after}"
    assert std_after.item() < std_before.item(), f"grad std did not improve: {std_before} -> {std_after}"
    # Fast marching should not move the level set at all
    drift = sdf_reinit_drift(phi_noisy, phi_reinit, dx)
    assert drift < 0.1 * dx, f"interface drift too large: {drift}"