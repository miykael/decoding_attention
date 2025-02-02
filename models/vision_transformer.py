"""
Vision Transformer Models

This module implements Vision Transformer (ViT) variants for processing image data.
The base model, VisionTransformerModel, implements the core structure:
  - Patch embedding that splits images into fixed-size patches
  - Positional embeddings added to patch tokens
  - Transformer encoder to process patch sequences
  - Transformer decoder with cross-attention to reconstruct the image

The implementation is designed to be clear and didactic, showing how:
  - Images can be processed as sequences of patches
  - Positional information helps maintain spatial relationships
  - Cross-attention allows the decoder to selectively focus on encoded patches
"""

import torch
import torch.nn as nn
from torch import Tensor

from config import (
    VISION_TRANSFORMER_IMAGE_SIZE,
    VISION_TRANSFORMER_PATCH_SIZE,
    VISION_TRANSFORMER_EMBED_DIM,
    VISION_TRANSFORMER_ENCODER_DEPTH,
    VISION_TRANSFORMER_DECODER_DEPTH,
    VISION_TRANSFORMER_NUM_HEADS,
    VISION_TRANSFORMER_DROPOUT,
)


class PatchEmbedding(nn.Module):
    """
    Splits the input image into patches and projects each patch to an embedding vector.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        # Use a Conv2d with kernel_size=stride=patch_size to extract non-overlapping patches.
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image tensor of shape [B, C, H, W].
        Returns:
            Tensor of shape [B, N, embed_dim] where N is the number of patches.
        """
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        B, E, H_p, W_p = x.shape
        # Flatten spatial dimensions and transpose to [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformerModel(nn.Module):
    """
    Vision Transformer that processes images using patch embeddings and transformer blocks.
    Uses PyTorch's built-in transformer modules for clarity and efficiency.

    The model processes an input of shape [B, 3, H, W] through these steps:
    1. Split image into fixed-size patches and embed them
    2. Add learned positional embeddings
    3. Process with transformer encoder
    4. Decode with transformer decoder using learned queries
    5. Project patches back to pixel space and reconstruct the image
    """

    def __init__(self) -> None:
        super().__init__()
        self.image_size = VISION_TRANSFORMER_IMAGE_SIZE  # e.g., (128, 128)
        self.patch_size = VISION_TRANSFORMER_PATCH_SIZE  # Size of each patch
        self.embed_dim = VISION_TRANSFORMER_EMBED_DIM  # Embedding dimension
        H, W = self.image_size
        # Compute number of patches (assumes H, W are divisible by patch_size)
        self.num_patches = (H // self.patch_size) * (W // self.patch_size)

        # Patch embedding and positional encoding
        self.patch_embed = PatchEmbedding(
            in_channels=3, embed_dim=self.embed_dim, patch_size=self.patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=VISION_TRANSFORMER_NUM_HEADS,
            dropout=VISION_TRANSFORMER_DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=VISION_TRANSFORMER_ENCODER_DEPTH
        )

        # Decoder queries and transformer decoder
        self.dec_queries = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim)
        )
        nn.init.trunc_normal_(self.dec_queries, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=VISION_TRANSFORMER_NUM_HEADS,
            dropout=VISION_TRANSFORMER_DROPOUT,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=VISION_TRANSFORMER_DECODER_DEPTH
        )

        # Final projection to patch pixels
        self.patch_dim = self.patch_size * self.patch_size * 3
        self.patch_project = nn.Linear(self.embed_dim, self.patch_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Process input image through the Vision Transformer.

        Args:
            x: Input image tensor of shape [B, 3, H, W].
        Returns:
            Reconstructed output image tensor of shape [B, 3, H, W].
        """
        B = x.size(0)
        # 1. Patch embedding and add positional information
        x_patches = self.patch_embed(x)  # [B, num_patches, embed_dim]
        x_patches = x_patches + self.pos_embed

        # 2. Process patches with the Transformer encoder
        memory = self.encoder(x_patches)  # [B, num_patches, embed_dim]

        # 3. Prepare and expand learned decoder queries
        queries = self.dec_queries.expand(B, -1, -1)

        # 4. Process with the Transformer decoder (cross-attending to encoder memory)
        dec_out = self.decoder(queries, memory)  # [B, num_patches, embed_dim]

        # 5. Project decoder outputs to pixel predictions per patch
        patch_preds = self.patch_project(dec_out)  # [B, num_patches, patch_dim]

        # 6. Reassemble patches into the full image
        output = self._assemble_patches(
            patch_preds, self.image_size[0], self.image_size[1]
        )
        return output

    def _assemble_patches(self, patch_preds: Tensor, H: int, W: int) -> Tensor:
        """
        Reshapes the predicted patches into a full image.

        Args:
            patch_preds: Tensor of shape [B, num_patches, patch_dim].
            H: Image height.
            W: Image width.
        Returns:
            Tensor of shape [B, 3, H, W].
        """
        B = patch_preds.size(0)
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size

        # Reshape patches into grid: [B, grid_h, grid_w, patch_size, patch_size, 3]
        patch_preds = patch_preds.view(
            B, grid_h, grid_w, self.patch_size, self.patch_size, 3
        )
        # Rearrange dimensions to form the image
        output = patch_preds.permute(0, 5, 1, 3, 2, 4).contiguous()
        output = output.view(B, 3, H, W)
        return output
