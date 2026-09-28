import torch

from nucleus.utils.losses import field_gradient_loss, sdf_sign_bce_loss


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


def test_sdf_sign_bce_loss_penalizes_wrong_sign_far_more_than_correct_sign():
    # Same |sdf| magnitude in both cases (so an L1 loss against a shifted target
    # would look identical) -- BCE cares about the sign, not just the magnitude, so
    # a confidently-wrong-sign prediction should cost far more than a confidently
    # -correct one.
    target_sdf = torch.tensor([[[[-3.0]]]])  # liquid
    correct = sdf_sign_bce_loss(torch.tensor([[[[-3.0]]]]), target_sdf, vapor_weight=1.0)
    wrong = sdf_sign_bce_loss(torch.tensor([[[[3.0]]]]), target_sdf, vapor_weight=1.0)
    assert wrong.item() > 10 * correct.item()


def test_sdf_sign_bce_loss_small_for_confidently_correct_sign():
    target_sdf = torch.tensor([[[[-5.0, 5.0]]]])
    pred_sdf = torch.tensor([[[[-5.0, 5.0]]]])
    assert sdf_sign_bce_loss(pred_sdf, target_sdf, vapor_weight=1.0).item() < 0.01


def test_sdf_sign_bce_loss_vapor_weight_upweights_missed_vapor():
    # A missed vapor pixel (predicted liquid, actually vapor) should cost more as
    # vapor_weight increases; a missed liquid pixel is unaffected by vapor_weight.
    target_sdf = torch.tensor([[[[5.0]]]])   # vapor
    pred_sdf = torch.tensor([[[[-5.0]]]])    # predicted liquid: wrong

    low_weight = sdf_sign_bce_loss(pred_sdf, target_sdf, vapor_weight=1.0)
    high_weight = sdf_sign_bce_loss(pred_sdf, target_sdf, vapor_weight=20.0)
    assert high_weight.item() > low_weight.item()
