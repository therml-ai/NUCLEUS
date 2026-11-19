from typing import Dict, Tuple

import torch
from boiling_data.flashx.normalizer import NormalizerWrapper
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch

PredictionTargetPairs = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


def normalize_forecast_batch(normalizer: NormalizerWrapper, batch: FlashXForecastBatch) -> FlashXForecastBatch:
    return FlashXForecastBatch(normalizer.normalize(batch.input), normalizer.normalize(batch.target))


def pooled_l1_loss(pairs: PredictionTargetPairs) -> torch.Tensor:
    """The mean absolute error over the cells of every predicted grid together, so
    fields on grids of different shapes, such as cell centers and faces, weigh by
    their number of samples."""
    total_abs_error = sum((prediction - goal).abs().sum() for prediction, goal in pairs.values())
    total_elements = sum(prediction.numel() for prediction, _ in pairs.values())
    return total_abs_error / total_elements


def per_field_mae(pairs: PredictionTargetPairs, prefix: str) -> Dict[str, torch.Tensor]:
    return {f"{prefix}/mae_{name}": (prediction - goal).abs().mean().detach() for name, (prediction, goal) in pairs.items()}
