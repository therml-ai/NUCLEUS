from contextlib import contextmanager
from typing import Iterator, Sequence, Tuple, Union
from unittest import mock

import natten
import torch


def reference_neighborhood_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
    dilation: Union[int, Sequence[int]] = 1,
    is_causal: bool = False,
    scale: float = None,
    **backend_options,
) -> torch.Tensor:
    """Plain PyTorch neighborhood attention on ``[batch, *spatial, heads, head_dim]``
    tensors, matching ``natten.na2d`` / ``natten.na3d`` for stride 1, no dilation and
    no causal masking. It is differentiable on any device, unlike NATTEN on CPU."""
    spatial_shape = tuple(query.shape[1:-2])
    if _as_tuple(stride, len(spatial_shape)) != (1,) * len(spatial_shape) or is_causal:
        raise NotImplementedError("reference neighborhood attention supports stride 1 without causal masking")
    if _as_tuple(dilation, len(spatial_shape)) != (1,) * len(spatial_shape):
        raise NotImplementedError("reference neighborhood attention does not support dilation")
    batch, heads, head_dim = query.shape[0], query.shape[-2], query.shape[-1]
    scale = head_dim ** -0.5 if scale is None else scale
    neighbors = _neighbor_indices(spatial_shape, _as_tuple(kernel_size, len(spatial_shape)), query.device)
    # batched matmuls on contiguous [batch, heads, position, neighbor, head_dim]
    # tensors; einsum over the gathered neighbors miscompiles under torch.compile
    query_rows = query.reshape(batch, -1, heads, head_dim).permute(0, 2, 1, 3).unsqueeze(-2)
    key_neighbors = _gather_neighbors(key, neighbors, batch, heads, head_dim)
    value_neighbors = _gather_neighbors(value, neighbors, batch, heads, head_dim)
    scores = torch.matmul(query_rows, key_neighbors.transpose(-1, -2)) * scale
    weights = torch.softmax(scores.float(), dim=-1).to(value.dtype)
    output = torch.matmul(weights, value_neighbors).squeeze(-2)
    # contiguous like NATTEN's output, which callers view into other shapes
    return output.permute(0, 2, 1, 3).reshape(query.shape).contiguous()


def _gather_neighbors(tensor: torch.Tensor, neighbors: torch.Tensor, batch: int, heads: int, head_dim: int) -> torch.Tensor:
    flat = tensor.reshape(batch, -1, heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return flat[:, :, neighbors].contiguous()


def _as_tuple(value: Union[int, Sequence[int]], length: int) -> Tuple[int, ...]:
    return (value,) * length if isinstance(value, int) else tuple(value)


def _window_starts(length: int, kernel: int, device: torch.device) -> torch.Tensor:
    # NATTEN shifts a border window inward rather than padding it, so every query
    # sees exactly `kernel` neighbors
    return (torch.arange(length, device=device) - kernel // 2).clamp(0, length - kernel)


def _neighbor_indices(spatial_shape: Tuple[int, ...], kernel_sizes: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """[num_positions, num_neighbors] flat indices of every position's neighbors."""
    num_axes = len(spatial_shape)
    flat_strides = torch.tensor(spatial_shape[1:] + (1,)).flip(0).cumprod(0).flip(0).tolist()
    indices = torch.zeros((1,) * (2 * num_axes), dtype=torch.long, device=device)
    for axis, (length, kernel, flat_stride) in enumerate(zip(spatial_shape, kernel_sizes, flat_strides)):
        window = _window_starts(length, kernel, device)[:, None] + torch.arange(kernel, device=device)
        shape = [1] * (2 * num_axes)
        shape[axis], shape[num_axes + axis] = length, kernel
        indices = indices + (window * flat_stride).reshape(shape)
    return indices.reshape(-1, int(torch.tensor(kernel_sizes).prod()))


@contextmanager
def reference_natten() -> Iterator[None]:
    """Route natten.na2d / natten.na3d through the reference implementation."""
    with mock.patch.object(natten, "na2d", reference_neighborhood_attention), \
            mock.patch.object(natten, "na3d", reference_neighborhood_attention):
        yield
