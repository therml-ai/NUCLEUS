import torch


def _eikonal_metrics(
    gradient: torch.Tensor, interface_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    residual = (torch.linalg.vector_norm(gradient, dim=0) - 1).square()
    interface_count = interface_mask.sum(dim=(-2, -1))
    return {
        "eikonal_loss": residual.mean(dim=(-2, -1)),
        "interface_eikonal_loss": torch.where(
            interface_count > 0,
            (residual * interface_mask).sum(dim=(-2, -1))
            / interface_count.clamp_min(1),
            float("nan"),
        ),
    }


def sdf_frame_metrics(
    prediction: torch.Tensor, target: torch.Tensor, dx: float, dy: float
) -> dict[str, torch.Tensor]:
    """Compute metrics for SDF tensors with shape (time, height, width)."""
    prediction_mask = prediction > 0
    target_mask = target > 0
    intersection = (prediction_mask & target_mask).sum(dim=(-2, -1))
    union = (prediction_mask | target_mask).sum(dim=(-2, -1))
    interface_mask = target.abs() <= 3 * max(dx, dy)
    interface_count = interface_mask.sum(dim=(-2, -1))
    prediction_gradient = torch.stack(torch.gradient(
        prediction, spacing=(dy, dx), dim=(-2, -1)
    ))
    target_gradient = torch.stack(torch.gradient(
        target, spacing=(dy, dx), dim=(-2, -1)
    ))
    gradient_error = torch.linalg.vector_norm(
        prediction_gradient - target_gradient, dim=0
    )
    interface_error = (gradient_error * interface_mask).sum(dim=(-2, -1))
    return {
        **_eikonal_metrics(prediction_gradient, interface_mask),
        "sdf_mae": (prediction - target).abs().mean(dim=(-2, -1)),
        "sdf_mse": (prediction - target).square().mean(dim=(-2, -1)),
        "bubble_iou": torch.where(union > 0, intersection / union.clamp_min(1), 1.0),
        "interface_gradient_error": torch.where(
            interface_count > 0,
            interface_error / interface_count.clamp_min(1),
            float("nan"),
        ),
    }
