import random
from typing import Callable, Optional

import torch
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch


def sample_num_unrolls(max_unrolls: int, probability: float, generator: Optional[random.Random] = None) -> int:
    """No unrolling with probability 1 - probability, otherwise a number of unrolls
    drawn uniformly from [1, max_unrolls]."""
    if max_unrolls < 0 or not 0.0 <= probability <= 1.0:
        raise ValueError(f"need max_unrolls >= 0 and probability in [0, 1], not {max_unrolls} and {probability}")
    generator = generator or random
    if max_unrolls == 0 or generator.random() >= probability:
        return 0
    return generator.randint(1, max_unrolls)


def push_forward(
    predict: Callable[[FlashXBatch], FlashXBatch],
    batch: FlashXForecastBatch,
    num_unrolls: int,
    target_timesteps: int,
) -> FlashXForecastBatch:
    """The input a model reaches after predicting num_unrolls target windows past
    batch.input, and the target window that follows it. The target holds every
    unrolled window, as FlashXForecastDataset's unroll_steps lays it out. The
    unrolls run without gradients, so the model is trained on its own predictions
    as inputs without backpropagating through the rollout. predict maps an input
    window to the next target_timesteps frames."""
    history = batch.input
    with torch.no_grad():
        for _ in range(num_unrolls):
            history = history.extend(predict(history)).tail_time_window(history.num_timesteps)
    target = batch.target.head_time_window((num_unrolls + 1) * target_timesteps)
    return FlashXForecastBatch(history, target.tail_time_window(target_timesteps))
