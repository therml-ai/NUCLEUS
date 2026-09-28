import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel_2d(kernel_size: int, sigma: float, dtype=torch.float32) -> torch.Tensor:
    """Separable 2D Gaussian convolution kernel, shape ``(1, 1, kernel_size,
    kernel_size)``, normalized to sum to 1. Summing to 1 preserves constants, so
    smoothing commutes with an affine (un)normalization and leaves a downstream curl or
    gradient unchanged by any DC shift."""
    coords = torch.arange(kernel_size, dtype=dtype) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d).reshape(1, 1, kernel_size, kernel_size)


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size: int, sigma: float):
        super().__init__()
        # _is_identity is a Python constant, so torch.compile specializes the branch and
        # does not trace the conv when the filter is disabled.
        self._is_identity = kernel_size < 3
        if self._is_identity:
            return
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.padding = kernel_size // 2
        self.register_buffer("kernel", gaussian_kernel_2d(kernel_size, sigma), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_identity:
            return x
        *leading, height, width = x.shape
        flat = x.reshape(-1, 1, height, width)
        flat = F.pad(flat, (self.padding,) * 4, mode="reflect")
        smoothed = F.conv2d(flat, self.kernel.to(flat.dtype))
        return smoothed.reshape(*leading, height, width)