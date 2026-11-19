import torch

from nucleus.utils.sdf_metrics import sdf_frame_metrics


def test_identical_fields_have_perfect_metrics():
    target = torch.tensor([[[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]])
    metrics = sdf_frame_metrics(target, target, dx=1.0, dy=2.0)
    assert metrics['sdf_mae'].item() == 0
    assert metrics['sdf_mse'].item() == 0
    assert metrics['bubble_iou'].item() == 1
    assert metrics['interface_gradient_error'].item() == 0
    assert metrics['eikonal_loss'].item() == 0
    assert metrics['interface_eikonal_loss'].item() == 0


def test_disjoint_bubbles_and_known_mae():
    target = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
    metrics = sdf_frame_metrics(-target, target, dx=1.0, dy=1.0)
    assert metrics['sdf_mae'].item() == 2
    assert metrics['sdf_mse'].item() == 4
    assert metrics['bubble_iou'].item() == 0
    assert metrics['interface_gradient_error'].item() == 4


def test_empty_masks_and_missing_interface():
    target = torch.full((2, 3, 4), -100.0)
    metrics = sdf_frame_metrics(target, target, dx=1.0, dy=1.0)
    assert torch.all(metrics['bubble_iou'] == 1)
    assert torch.isnan(metrics['interface_gradient_error']).all()
    assert torch.all(metrics['eikonal_loss'] == 1)
    assert torch.isnan(metrics['interface_eikonal_loss']).all()


def test_eikonal_loss_uses_physical_spacing_and_squared_residual():
    target = torch.zeros(2, 3, 4)
    prediction = torch.arange(4).float()[None, None, :].expand_as(target) * 6
    metrics = sdf_frame_metrics(prediction, target, dx=2.0, dy=5.0)
    assert torch.allclose(metrics['eikonal_loss'], torch.full((2,), 4.0))
    assert torch.allclose(metrics['interface_eikonal_loss'], torch.full((2,), 4.0))


def test_anisotropic_spacing_matches_linear_gradient_error():
    target = torch.zeros(1, 3, 4)
    prediction = torch.arange(4).float()[None, None, :].expand_as(target) * 2
    metrics = sdf_frame_metrics(prediction, target, dx=2.0, dy=5.0)
    assert torch.allclose(metrics['interface_gradient_error'], torch.ones(1))


def test_mse_averages_squared_errors_per_frame():
    target = torch.zeros(2, 2, 2)
    prediction = torch.tensor([[[0., 2.], [0., 2.]], [[1., 3.], [1., 3.]]])
    metrics = sdf_frame_metrics(prediction, target, dx=1.0, dy=1.0)
    assert torch.allclose(metrics['sdf_mse'], torch.tensor([2.0, 5.0]))
