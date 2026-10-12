import torch
import torch.nn as nn
import torch.nn.functional as F

# Parameters are stored in float32 so the optimizer's small updates are not rounded
# away. Any computation casts to the dtype of each layer's input, e.g. bfloat16, for the compute.

class CastLinear(nn.Linear):
    """nn.Linear computed in its input's dtype."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(input.dtype)
        return F.linear(input, self.weight.to(input.dtype), bias)


class CastRMSNorm(nn.RMSNorm):
    """nn.RMSNorm computed in its input's dtype."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = None if self.weight is None else self.weight.to(input.dtype)
        return F.rms_norm(input, self.normalized_shape, weight, self.eps)
