import torch

from nucleus.physics.sdf import band_mask


def _vertical_sdf(height, width, dx, dy):
    """Signed distance for a vertical interface: liquid (sdf < 0) left, vapor right.
    The normal is +x everywhere."""
    x = (torch.arange(width, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(height, dtype=torch.float64) + 0.5) * dy
    _, grid_x = torch.meshgrid(y, x, indexing="ij")
    x0 = x[width // 2] + 0.5 * dx
    return grid_x - x0


def test_band_mask_widens_with_width():
    dx = dy = 1.0 / 32
    sdf = _vertical_sdf(48, 48, dx, dy)
    narrow = int(band_mask(sdf, 1 * dx).sum())
    wide = int(band_mask(sdf, 3 * dx).sum())
    assert wide > narrow
