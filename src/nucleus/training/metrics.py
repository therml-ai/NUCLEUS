from typing import Dict

import torch


@torch.no_grad()
def prediction_metrics(prediction: torch.Tensor, target: torch.Tensor, prefix: str) -> Dict[str, torch.Tensor]:
    return {
        f"{prefix}/mae_loss": torch.nn.functional.l1_loss(prediction, target),
        f"{prefix}/mse_loss": torch.nn.functional.mse_loss(prediction, target),
        f"{prefix}/absmax_error": (prediction - target).abs().max(),
        f"{prefix}/pred_mean": prediction.mean(),
        f"{prefix}/pred_std": prediction.std(),
        f"{prefix}/target_mean": target.mean(),
        f"{prefix}/target_std": target.std(),
    }
