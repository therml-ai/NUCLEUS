import torch
import torch.nn as nn
from einops import rearrange

from nucleus.data.batching import CollatedBatch
from nucleus.layers import BubbleformerAxialAttentionBlock, BubbleformerAttentionBlock, HMLPEmbed, HMLPDebed, FiLMMLP
from ._api import register_model

__all__ = ["BubbleformerViT"]


class SpaceTimeBlock(nn.Module):
    """
    Factored spacetime block with temporal attention followed by axial attention
    Args:
        embed_dim (int): Number of features in the input tensor
        num_heads (int): Number of attention heads
        attn_scale (bool): Whether to use attention scaling
        feat_scale (bool): Whether to use feature scaling
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_scale: bool = True,
        feat_scale: bool = True,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.temporal = BubbleformerAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_scale=attn_scale,
        )

        self.spatial = BubbleformerAxialAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_scale=attn_scale,
            feat_scale=feat_scale,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x) -> torch.Tensor:
        """
        Args: 
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C, H, W)
        """
        _, t, _, _, _ = x.shape

        # First do temporal attention
        x = self.temporal(x)    # (B, T, emb, H, W)

        # Now do spatial attention
        x = rearrange(x, "b t emb h w -> (b t) emb h w")        # BT sequences
        x = self.spatial(x)                                 # A spatial encoder block
        x = rearrange(x, "(b t) emb h w -> b t emb h w", t=t)

        return x    # (B, T, emb, H, W)


@register_model("bubbleformer_vit")
class BubbleformerViT(nn.Module):
    """
    Model that interweaves spatial and temporal attention blocks. Temporal attention
    acts only on the time dimension.

    Args:
        fields (int): Number of fields
        time_window (int): Number of time steps
        patch_size (int): Size of the square patch
        embed_dim (int): Dimension of the embedding
        num_heads (int): Number of attention heads
        processor_blocks (int): Number of processor blocks
        attn_scale (bool): Whether to use attention scaling
        feat_scale (bool): Whether to use feature scaling
    """
    def __init__(
        self,
        input_fields: int = 3,
        output_fields: int = 3,
        time_window: int = 12,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        processor_blocks: int = 12,
        attn_scale: bool = True,
        feat_scale: bool = True,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        # Hierarchical Patch Embedding
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )
        # Factored spacetime block with (space/time axial attention)
        self.blocks = nn.ModuleList(
            [
                SpaceTimeBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_scale=attn_scale,
                    feat_scale=feat_scale,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(processor_blocks)
            ]
        )
        # Patch Debedding
        self.debed = HMLPDebed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=output_fields
        )

    def forward(self, batch: CollatedBatch) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C, H, W)
        """
        x = batch.input
        _, t, _, _, _ = x.shape

        # Encode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)

        # Process
        for blk in self.blocks:
            # x = cp.checkpoint(blk, x, use_reentrant=False)
            x = blk(x)

        # Decode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)

        return x  # Temporal bundling (B, T, C, H, W)


@register_model("bubbleformer_film_vit")
class BubbleformerFilmViT(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) is an expressive and lightweight way to 
    condition neural networks using external information (like fluid parameters).
    This model uses FiLM conditioning on the embeddings and the blocks.
    Args:
        input_fields (int): Number of input fields
        output_fields (int): Number of output fields
        time_window (int): Number of time steps
        patch_size (int): Size of the square patch
        embed_dim (int): Dimension of the embedding
        num_heads (int): Number of attention heads
        processor_blocks (int): Number of processor blocks
        attn_scale (bool): Whether to use attention scaling
        feat_scale (bool): Whether to use feature scaling
        num_fluid_params (int): Number of fluid parameters for conditioning
    """
    def __init__(
        self,
        input_fields: int = 3,
        output_fields: int = 3,
        time_window: int = 12,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        processor_blocks: int = 12,
        attn_scale: bool = True,
        feat_scale: bool = True,
        num_fluid_params: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )

        self.film_embed = FiLMMLP(num_fluid_params, embed_dim)

        self.blocks = nn.ModuleList([
            SpaceTimeBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_scale=attn_scale,
                feat_scale=feat_scale,
                mlp_ratio=mlp_ratio,
            )
            for i in range(processor_blocks)
        ])

        self.debed = HMLPDebed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=output_fields
        )

    def forward(self, batch: CollatedBatch) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        fluid_params: (B, num_fluid_params)
        """
        x = batch.input
        fluid_params = batch.fluid_params_tensor
        B, T, _, _, _ = x.shape

        # Encode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed(x)
        x = rearrange(x, "(b t) c h w -> b t h w c", t=T)

        # Apply FiLM conditioning on the embeddings
        x = self.film_embed(x, fluid_params)  # (B, T, H, W, C)

        x = rearrange(x, "b t h w c -> b t c h w")

        # Process with FiLM-modulated blocks
        # for blk, film in zip(self.blocks, self.film_blocks):
        for blk in self.blocks:
            x = blk(x)

        # Decode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)
        return x
