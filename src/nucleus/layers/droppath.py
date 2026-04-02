import torch
import torch.nn as nn
import random
from typing import Callable

class DropPath(nn.Module):
    r"""based on stochastic depth implementation from pytorch vision"""
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival_rate = 1.0 - self.drop_prob
        size = [x.shape[0]] + [1] * (x.ndim - 1)
        noise = torch.empty(size, dtype=x.dtype, device=x.device)
        noise = noise.bernoulli_(survival_rate)
        if survival_rate > 0.0:
            noise.div_(survival_rate)
        return x * noise