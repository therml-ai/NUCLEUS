import torch
import torch.nn as nn

class DropPath(nn.Module):
    r"""based on stochastic depth implementation from pytorch vision"""
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob
        # _is_identity and registering buffer prevent recompilation if using torch.compile
        self._is_identity = drop_prob == 0.0
        self.register_buffer("survival_rate", torch.tensor(1.0 - drop_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self._is_identity:
            return x
        noise = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(self.survival_rate).div_(self.survival_rate)
        return x * noise