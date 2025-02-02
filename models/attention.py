"""
Models for Sequence Tasks Using Attention

This file defines several model variants. The base class,
StandardAttentionModel, encapsulates the common structure:
  - an input 1x1 convolution to embed a 1-channel input into a hidden space,
  - an attention block (defined in blocks.py),
  - an output 1x1 convolution to map back to one channel.

Derived classes override a "hook" called modify_features() so that
additional processing (e.g. adding positional encoding or binarization)
can be applied without duplicating the full forward pass. In cases
where the built-in structure is very different (e.g. residual scaling in
EnhancedAttentionModel), the forward() method is overridden completely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import SelfAttentionLayer, MultiHeadSelfAttention
from models.encoding import sinusoidal_positional_encoding


class StandardAttentionModel(nn.Module):
    """
    Standard Attention Model that defines the common conv → attention → conv workflow.

    This model is designed so that derived classes can change how the features are modified
    before entering the attention block by overriding modify_features().
    """

    def __init__(
        self, hidden_channels: int = 64, attention_block: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        # Input projection: map 1 channel to hidden_channels.
        self.input_conv = nn.Conv1d(1, hidden_channels, kernel_size=1)
        # Output projection: map from hidden_channels back to 1 channel.
        self.output_conv = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        # Use the provided attention block or default to SelfAttentionLayer.
        self.attention = attention_block or SelfAttentionLayer(
            in_dim=hidden_channels, out_dim=hidden_channels, key_dim=hidden_channels
        )

    def modify_features(self, features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Hook method to modify the projected features before feeding into attention.
        By default, returns the features unchanged.
        """
        return features

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_qkv: bool = False,
    ):
        # x shape: (batch, 1, seq_len)
        # Project input using a 1x1 convolution and apply ReLU activation.
        features = F.relu(self.input_conv(x))
        # Optionally modify features (e.g., adding positional encoding in derived models).
        features = self.modify_features(features, x)
        # Apply the attention block.
        attn_out = self.attention(
            features,
            return_attention=return_attention,
            return_qkv=return_qkv,
        )
        if return_attention or return_qkv:
            # attn_out is a tuple: (output, attention, (...optional...))
            output = self.output_conv(attn_out[0])
            extra_info = attn_out[1:]
            return output, extra_info
        else:
            return self.output_conv(attn_out)


class AttentionModelWithPE(StandardAttentionModel):
    """
    Attention Model with Positional Encoding.

    Inherits from StandardAttentionModel and overrides the feature modification
    hook to add sinusoidal positional encoding.
    """

    def __init__(self, hidden_channels: int = 64) -> None:
        super().__init__(hidden_channels=hidden_channels)
        # Extra convolution to process positional encoding features.
        self.pe_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)

    def modify_features(self, features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len = x.size()
        # Generate sinusoidal positional encoding with shape (seq_len, 1, hidden_channels).
        pe = sinusoidal_positional_encoding(seq_len, d_model=self.hidden_channels)
        # Convert to float, adjust dimensions: (seq_len, 1, hidden_channels) -> (batch, hidden_channels, seq_len)
        pe_tensor = (
            pe.float()
            .to(x.device)
            .squeeze(
                1
            )  # from (seq_len, 1, hidden_channels) to (seq_len, hidden_channels)
            .transpose(0, 1)  # to (hidden_channels, seq_len)
            .unsqueeze(0)  # add batch dimension -> (1, hidden_channels, seq_len)
            .expand(batch_size, -1, -1)
        )
        # Process the positional encoding via an extra convolution.
        pe_features = self.pe_conv(pe_tensor)
        # Add the processed positional encoding to the original features and apply ReLU.
        return F.relu(features + pe_features)


class BitNetModel(StandardAttentionModel):
    """
    BitNet Model that applies feature binarization.

    Overrides the forward pass so that ReLU is applied before binarizing the features.
    Positive activations become 1 and negative ones become -1.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input and apply ReLU activation.
        features = F.relu(self.input_conv(x))
        # Binarize features: positive values become 1, negatives become -1.
        features = torch.sign(features)
        # Apply the attention block.
        attn_out = self.attention(features)
        # Map features back to 1 channel.
        return self.output_conv(attn_out)


class MultiHeadAttentionModel(StandardAttentionModel):
    """
    Multi-head Attention Model with Positional Encoding.

    Uses multi-head self-attention (via MultiHeadSelfAttention) instead of single-head attention.
    """

    def __init__(self, n_heads: int = 8, hidden_channels: int = 64) -> None:
        # Initialize using MultiHeadSelfAttention as the attention block.
        attention_block = MultiHeadSelfAttention(
            in_dim=hidden_channels,
            out_dim=hidden_channels,
            key_dim=hidden_channels,
            n_heads=n_heads,
        )
        super().__init__(
            hidden_channels=hidden_channels, attention_block=attention_block
        )
        self.pe_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)

    def modify_features(self, features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len = x.size()
        # Generate sinusoidal positional encoding.
        pe = sinusoidal_positional_encoding(seq_len, d_model=self.hidden_channels)
        pe_tensor = (
            pe.float()
            .to(x.device)
            .squeeze(1)
            .transpose(0, 1)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        pe_features = self.pe_conv(pe_tensor)
        return F.relu(features + pe_features)


class EnhancedAttentionModel(StandardAttentionModel):
    """
    Enhanced Attention Model with residual scaling and stochastic depth.

    Overrides the forward pass to include:
         x_out = x_in + stochastic_depth(scale * attention_out)
    which helps stabilize training.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        stochastic_depth_prob: float = 0.1,
        layer_scale_init_value: float = 1e-5,
    ) -> None:
        super().__init__(hidden_channels=hidden_channels)
        # Learnable scaling parameter for the attention branch.
        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones(hidden_channels), requires_grad=True
        )
        # Use dropout for stochastic depth if probability > 0.
        self.stochastic_depth = (
            nn.Dropout(stochastic_depth_prob)
            if stochastic_depth_prob > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection with ReLU activation.
        x_in = F.relu(self.input_conv(x))
        # Compute attention output.
        attn_out = self.attention(x_in)
        # Apply layer scaling and stochastic depth to the attention branch.
        scaled_attn = self.stochastic_depth(
            self.layer_scale.unsqueeze(0).unsqueeze(2) * attn_out
        )
        # Residual connection: add scaled attention output to the input features.
        x_out = x_in + scaled_attn
        return self.output_conv(x_out)
