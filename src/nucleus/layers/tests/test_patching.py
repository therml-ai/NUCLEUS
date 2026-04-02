import pytest
import torch
from nucleus.layers import HMLPDebed, HMLPEmbed

@pytest.mark.parametrize("patch_size", [4, 8, 16, 32])
@pytest.mark.parametrize("embed_dim", [192, 384, 768, 1024])
def test_patching_preserve_spatial(patch_size, embed_dim):
    """
    Test Axial ViT Patching using Hierarchical Conv2d
    """
    in_channels = 4
    spatial_dims = 64, 64
    embed = HMLPEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
    debed = HMLPDebed(patch_size=patch_size, out_channels=in_channels, embed_dim=embed_dim)

    x = torch.randn(1, in_channels, *spatial_dims) # (B, C, H, W)
    height_patches = spatial_dims[0] // patch_size
    width_patches = spatial_dims[1] // patch_size
    y = embed(x)
    z = debed(y)

    assert y.shape == (1, embed_dim, height_patches, width_patches) and z.shape == x.shape
