import math
import torch
import torch.nn as nn
import einops

@torch.compile(fullgraph=True)
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

@torch.compile(fullgraph=True)
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
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.linear = nn.Linear(in_channels * patch_size ** 2, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b t (h p1) (w p2) c -> b t h w (c p1 p2)", p1=self.patch_size, p2=self.patch_size)
        x = self.linear(x)
        return x

class LinearDebed(nn.Module):
    def __init__(self, patch_size: int, out_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, out_channels * patch_size ** 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = einops.rearrange(x, "b t h w (c p1 p2) -> b t (h p1) (w p2) c", p1=self.patch_size, p2=self.patch_size)
        return x