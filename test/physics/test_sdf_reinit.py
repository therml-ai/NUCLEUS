import pytest
import torch
from nucleus.physics.sdf_reinit import (
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
def test_sussman_reinitializes_noisy_circle(noise_std):
    torch.manual_seed(0)

    phi_true, dx = make_circle_sdf()

    phi_noisy = phi_true + noise_std * torch.randn_like(phi_true)
    phi_reinit = sdf_reinit_sussman(
        phi_noisy,
        dx=dx,
        n_iter=100,
    )
    
    # Check |grad(phi)| stats improved toward 1
    mean_before, std_before = verify_sdf(phi_noisy, dx)
    mean_after, std_after = verify_sdf(phi_reinit, dx)
    err_before = torch.abs(mean_before - 1.0).item()
    err_after = torch.abs(mean_after - 1.0).item()
    assert err_after < err_before, f"mean |grad|-1 did not improve: {err_before} -> {err_after}"
    assert std_after.item() < std_before.item(), f"grad std did not improve: {std_before} -> {std_after}"
    # Reinitializing removes noise near the interface, so it is compared against the
    # true circle rather than the noisy input.
    near_interface = phi_true.abs() < 3 * dx
    error_before = (phi_noisy - phi_true)[near_interface].abs().mean()
    error_after = (phi_reinit - phi_true)[near_interface].abs().mean()
    assert error_after < error_before, f"near-interface error grew: {error_before} -> {error_after}"
    assert _wrong_phase_cells(phi_reinit, phi_true) <= _wrong_phase_cells(phi_noisy, phi_true)


def test_sussman_keeps_exact_sdf_interface():
    phi_true, dx = make_circle_sdf()
    phi_reinit = sdf_reinit_sussman(phi_true, dx=dx, n_iter=100)
    assert sdf_reinit_drift(phi_true, phi_reinit, dx) < 0.1 * dx
    assert _wrong_phase_cells(phi_reinit, phi_true) == 0


def _wrong_phase_cells(phi, phi_true):
    return ((phi > 0) != (phi_true > 0)).sum().item()


def test_sussman_holds_contact_angle_at_heater_wall():
    # a plane meeting the wall (row 0) at the contact angle is already a signed
    # distance whose wall slope is cos(angle), so it should not move
    angle = 45.0
    num_cells, dx = 64, 0.1
    centers = (torch.arange(num_cells, dtype=torch.float64) + 0.5) * dx
    y_centers, x_centers = torch.meshgrid(centers, centers, indexing="ij")
    radians = torch.deg2rad(torch.tensor(angle, dtype=torch.float64))
    plane = torch.cos(radians) * y_centers + torch.sin(radians) * (x_centers - 3.2)
    # the side walls copy their edges, which bends the plane there, so only the
    # columns far from them are checked
    by_heater = (slice(0, 8), slice(16, 48))
    with_angle = sdf_reinit_sussman(plane, dx, n_iter=20, contact_angle=angle)
    replicated = sdf_reinit_sussman(plane, dx, n_iter=20)
    error_with_angle = (with_angle - plane)[by_heater].abs().max()
    error_replicated = (replicated - plane)[by_heater].abs().max()
    assert error_with_angle < 1e-3 * dx
    assert error_replicated > 10 * error_with_angle


def test_sussman_only_redistances_cells_within_the_reinit_band():
    phi_true, dx = make_circle_sdf()
    # a level set twice as steep as a distance, so every cell is redistanced
    steep = 2 * phi_true
    reinit_band = 4 * dx
    banded = sdf_reinit_sussman(steep, dx=dx, n_iter=20, reinit_band=reinit_band)
    full = sdf_reinit_sussman(steep, dx=dx, n_iter=20)
    outside = steep.abs() >= reinit_band
    torch.testing.assert_close(banded[outside], steep[outside])
    # the cells the band's edge is too far to reach match the full redistancing
    deep_inside = steep.abs() < reinit_band - 2 * dx
    torch.testing.assert_close(banded[deep_inside], full[deep_inside])
    assert not torch.allclose(banded[~outside], steep[~outside])
