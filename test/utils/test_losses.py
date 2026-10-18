import torch

from nucleus.utils.losses import field_gradient_loss


def _fields(tensor_2d: torch.Tensor) -> torch.Tensor:
    # promote an (H, W) field to the (B, T, H, W, C) layout the loss expects
    return tensor_2d[None, None, :, :, None]


def test_field_gradient_loss_zero_for_identical_fields():
    torch.manual_seed(0)
    fields = torch.randn(2, 3, 8, 8, 3)
    assert field_gradient_loss(fields, fields).item() == 0.0


def test_field_gradient_loss_zero_for_constant_offset():
    # a constant offset has zero gradient everywhere, so it must not be penalized
    torch.manual_seed(0)
    target = torch.randn(2, 3, 8, 8, 3)
    pred = target + 5.0
    assert field_gradient_loss(pred, target).item() < 1e-6


def test_field_gradient_loss_matches_known_slope_difference():
    rows = torch.arange(6, dtype=torch.float32)
    cols = torch.arange(6, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")

    target = _fields(grid_y + grid_x)            # unit slope in both directions
    pred = _fields(3.0 * grid_y + 3.0 * grid_x)  # triple the slope

    # central-difference gradient of a linear ramp is exactly its slope, so the
    # per-direction L1 error is |3 - 1| = 2, summed over the two directions.
    expected = 2.0 + 2.0
    assert abs(field_gradient_loss(pred, target).item() - expected) < 1e-5
