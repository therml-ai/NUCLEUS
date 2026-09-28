import math
import torch
import torch.nn as nn
from typing import Tuple, Union
import einops

class HMLPEmbed(nn.Module):
    """
    Image to Patch Embedding using hierarchical Conv2d.
    It preserves the spatial ordering of the patches
    Args:
        patch_size (int): Size of the square patch
        in_channels (int): Number of input channels
        embed_dim (int): Dimension of the embedding
    """
    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.patch_size = patch_size
        num_layers = int(math.log2(patch_size))
        assert (num_layers - math.log2(patch_size)) == 0, "Patch size must be a power of 2"

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        layers = []
        conv_in = in_channels
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if num_layers == 1:
                conv_out = embed_dim
            else:
                conv_out = embed_dim if is_last else embed_dim // 4
            layers.append(
                nn.Conv2d(
                    in_channels=conv_in,
                    out_channels=conv_out,
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    dtype=torch.bfloat16
                )
            )
            layers.append(nn.InstanceNorm2d(conv_out, affine=True, dtype=torch.bfloat16))
            if not is_last:
                layers.append(nn.GELU())
            conv_in = conv_out
        self.in_proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, Emb, H_patches, W_patches)
        """
        x = self.in_proj(x.to(torch.bfloat16))
        return x.to(torch.float32)

class HMLPDebed(nn.Module):
    """
    Patch to Image De-bedding using hierarchical ConvTranspose2d.
    It takes a spatially ordered tensor of embedded patches and reconstructs the image
    Args:
        patch_size (int): Size of the square patch
        out_channels (int): Number of output channels
        embed_dim (int): Dimension of the embedding
    """
    def __init__(
        self,
        patch_size: int = 16,
        out_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.patch_size = patch_size
        num_layers = int(math.log2(patch_size))
        assert (num_layers - math.log2(patch_size)) == 0, "Patch size must be a power of 2"

        self.out_channels = out_channels
        self.embed_dim = embed_dim
        layers = []
        conv_in = embed_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if num_layers == 1:
                conv_out = out_channels
            else:
                conv_out = out_channels if is_last else embed_dim // 4
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=conv_in,
                    out_channels=conv_out,
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    dtype=torch.bfloat16
                )
            )
            if not is_last:
                layers.append(nn.InstanceNorm2d(conv_out, affine=True, dtype=torch.bfloat16))
                layers.append(nn.GELU())
            conv_in = conv_out

        self.out_proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, Emb, H_patches, W_patches)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        x = self.out_proj(x.to(torch.bfloat16))
        return x.to(torch.float32)

class LinearEmbed(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int, dtype: torch.dtype):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.linear = nn.Linear(in_channels * patch_size ** 2, embed_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b t (h p1) (w p2) c -> b t h w (c p1 p2)", p1=self.patch_size, p2=self.patch_size)
        x = self.linear(x)
        return x

class LinearDebed(nn.Module):
    def __init__(self, patch_size: int, out_channels: int, embed_dim: int, dtype: torch.dtype):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, out_channels * patch_size ** 2, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        x = self.linear(x)
        x = einops.rearrange(x, "b t h w (c p1 p2) -> b t (h p1) (w p2) c", p1=self.patch_size, p2=self.patch_size)
        assert x.shape[-3] == target_shape[0] and x.shape[-2] == target_shape[1]
        return x


class OverlappingPatchDebed(nn.Module):
    def __init__(self, patch_size: int, out_channels: int, embed_dim: int, dtype: torch.dtype, overlap: int = None, output_padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.overlap = patch_size // 2 if overlap is None else overlap
        # ConvTranspose2d output = (in - 1) * stride - 2 * padding + kernel +
        # output_padding. With kernel = patch_size + 2 * overlap, padding = overlap,
        # stride = patch_size, this is (in * patch_size) + output_padding: the
        # default restores LinearDebed's (in * patch_size) shape, while
        # output_padding=1 emits the nodal (H+1, W+1) grid (one extra node per axis).
        self.conv_transpose = nn.ConvTranspose2d(
            embed_dim,
            out_channels,
            kernel_size=patch_size + 2 * self.overlap,
            stride=patch_size,
            padding=self.overlap,
            output_padding=output_padding,
            bias=False,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        leading = x.shape[:-3]
        patch_h, patch_w, channels = x.shape[-3], x.shape[-2], x.shape[-1]
        x = x.reshape(-1, patch_h, patch_w, channels).permute(0, 3, 1, 2)
        x = self.conv_transpose(x)
        out_h, out_w = x.shape[-2], x.shape[-1]
        x = x.permute(0, 2, 3, 1).reshape(*leading, out_h, out_w, self.out_channels)
        assert x.shape[-3] == target_shape[0] and x.shape[-2] == target_shape[1]
        return x


def _local_fourier_coords(
    h: int, w: int, patch_shape: Tuple[int, int], num_freq_bands: int,
    device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Local Fourier coordinates in [-1, 1] within each patch, tiled across (H, W)."""
    assert h % patch_shape[0] == 0
    assert w % patch_shape[1] == 0
    patch_h = h // patch_shape[0]
    patch_w = w // patch_shape[1]
    local_ys = torch.linspace(-1, 1, patch_h, device=device, dtype=dtype)
    local_xs = torch.linspace(-1, 1, patch_w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(local_ys, local_xs, indexing="ij")
    grid_y = grid_y.repeat(patch_shape[0], patch_shape[1])
    grid_x = grid_x.repeat(patch_shape[0], patch_shape[1])
    freqs = 2 ** torch.arange(num_freq_bands, device=device, dtype=dtype) * math.pi
    y_angles = grid_y.unsqueeze(-1) * freqs
    x_angles = grid_x.unsqueeze(-1) * freqs
    return torch.cat([
        torch.sin(y_angles), torch.cos(y_angles),
        torch.sin(x_angles), torch.cos(x_angles),
    ], dim=-1)  # (H, W, 4 * num_freq_bands)


class AdaptiveDebed(nn.Module):
    """
    Inverts AdaptiveEmbed: upsamples from patch space back to an arbitrary
    target resolution using bilinear interpolation, then projects channels.
    Local Fourier coordinates are concatenated with the upsampled patch tokens
    before the projection, mirroring AdaptiveEmbed.

    The target spatial size (H, W) is passed at forward time since it is
    determined by the original input resolution, which varies at runtime.
    """
    def __init__(self, in_channels: int, out_channels: int, patch_shape: Tuple[int, int], dtype: torch.dtype, num_freq_bands: int = 4):
        super().__init__()
        self.patch_shape = patch_shape
        self.num_freq_bands = num_freq_bands
        self.linear = nn.Linear(in_channels + 4 * num_freq_bands, out_channels, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        leading = x.shape[:-3]
        h, w, c = x.shape[-3], x.shape[-2], x.shape[-1]

        flat = x.reshape(-1, h, w, c).permute(0, 3, 1, 2)
        flat = torch.nn.functional.interpolate(flat, size=target_shape, mode="bilinear", align_corners=False)
        flat = flat.permute(0, 2, 3, 1).reshape(*leading, *target_shape, c)

        coords = _local_fourier_coords(target_shape[0], target_shape[1], self.patch_shape, self.num_freq_bands, x.device, x.dtype)
        flat = torch.cat([flat, coords.expand(*leading, *target_shape, 4 * self.num_freq_bands)], dim=-1)

        return self.linear(flat)


class AdaptiveEmbed(nn.Module):
    """
    Breaks an arbitrary-resolution input into patches by average pooling to a
    fixed output shape. Patch size is implicitly (H / out_H, W / out_W) and
    adapts to whatever resolution is passed at runtime.

    Local Fourier coordinates are projected into embed space before pooling
    so that within-patch position information survives the averaging step.
    """
    def __init__(self, in_channels, out_channels, out_shape: Tuple[int, int], dtype: torch.dtype, num_freq_bands: int = 4):
        super().__init__()
        self.out_shape = out_shape
        self.num_freq_bands = num_freq_bands
        self.linear = nn.Linear(in_channels, out_channels, bias=False, dtype=dtype)
        self.coord_proj = nn.Linear(4 * num_freq_bands, out_channels, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-3]
        h, w = x.shape[-3], x.shape[-2]

        coords = _local_fourier_coords(h, w, self.out_shape, self.num_freq_bands, x.device, x.dtype)
        pos = self.coord_proj(coords)  # (H, W, out_channels)
        
        ll = self.linear(x) + pos

        # adaptive_avg_pool2d expects (N, C, H, W)
        c = ll.shape[-1]
        flat = ll.reshape(-1, h, w, c).permute(0, 3, 1, 2)
        flat = torch.nn.functional.adaptive_avg_pool2d(flat, self.out_shape)
        flat = flat.permute(0, 2, 3, 1)

        return flat.reshape(*leading, *self.out_shape, c)