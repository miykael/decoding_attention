"""
Positional Encoding Functions and Classes

This module provides functions for generating various types of positional encodings:
    - Linear encoding (increases linearly from 0 to 1)
    - Learnable encoding (trainable parameters)
    - Sinusoidal encoding (non-learnable using sine and cosine functions)
    - Rotary Positional Encoding (RoPE) for attention mechanisms
"""

import math
import torch
import torch.nn as nn
from config import PE_MIN_FREQ, PE_MAX_LENGTH, ROPE_BASE


def linear_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Generate a simple linear positional encoding.

    The encoding increases linearly from 0 to 1 across the sequence and is repeated across the feature dimension.

    Args:
        seq_len: Sequence length.
        d_model: Model/embedding dimension.

    Returns:
        Tensor of shape (1, seq_len, d_model) with the linear positional encodings.
    """
    # Generate linearly spaced values (shape: [seq_len, 1])
    pos = torch.linspace(0, 1, steps=seq_len).unsqueeze(1)
    # Repeat the values along the feature dimension (shape: [seq_len, d_model])
    pos_enc = pos.repeat(1, d_model)
    # Add a batch dimension: final shape (1, seq_len, d_model)
    return pos_enc.unsqueeze(0)


def learnable_positional_encoding(seq_len: int, d_model: int) -> nn.Parameter:
    """
    Generate learnable positional encodings.

    Initializes a trainable parameter with a normal distribution.

    Args:
        seq_len: Sequence length.
        d_model: Model/embedding dimension.

    Returns:
        A trainable nn.Parameter of shape (1, seq_len, d_model).
    """
    # Create a parameter with zeros and initialize it using a normal distribution.
    param = nn.Parameter(torch.zeros(1, seq_len, d_model))
    nn.init.normal_(param, mean=0, std=0.02)
    return param


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Generate sinusoidal positional encoding.

    Uses sine and cosine functions of different frequencies (following the original Transformer paper).

    Args:
        seq_len: Sequence length (clipped to PE_MAX_LENGTH if necessary).
        d_model: Model/embedding dimension (must be even).

    Returns:
        Tensor of shape (seq_len, 1, d_model) with sinusoidal positional encodings.

    Raises:
        ValueError: If d_model is not even.
    """
    if d_model % 2 != 0:
        raise ValueError(f"Model dimension {d_model} must be even")

    # Clip the sequence length to the maximum allowed value.
    seq_len = min(seq_len, PE_MAX_LENGTH)
    # Generate position indices (shape: [seq_len, 1])
    position = torch.arange(seq_len).unsqueeze(1)
    # Compute the scaling terms for sine and cosine (shape: [d_model/2])
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(PE_MIN_FREQ) / d_model)
    )
    # Initialize the positional encoding tensor.
    pos_enc = torch.zeros(seq_len, 1, d_model)
    # Apply sine to even indices and cosine to odd indices.
    pos_enc[:, 0, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 0, 1::2] = torch.cos(position * div_term)
    return pos_enc


def rotary_positional_encoding(
    x: torch.Tensor, dim: int, max_length: int = PE_MAX_LENGTH, base: int = ROPE_BASE
) -> torch.Tensor:
    """
    Apply Rotary Positional Encoding (RoPE) to an input tensor.

    Args:
        x: Input tensor with shape (..., seq_len, dim).
        dim: Feature dimension (must be even).
        max_length: Maximum sequence length (default: PE_MAX_LENGTH).
        base: Base for frequency computation (default: ROPE_BASE).

    Returns:
        Tensor with RoPE applied.

    Raises:
        ValueError: If dim is not even.
    """
    if dim % 2 != 0:
        raise ValueError(f"Feature dimension {dim} must be even for RoPE")

    # Compute half the dimension (each rotary pair uses two dimensions).
    half_dim = dim // 2
    # Calculate the inverse frequency for each dimension.
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim).float() / half_dim))
    # Create a tensor of positions and calculate angles.
    positions = torch.arange(max_length).float()
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    # Compute cosine and sine matrices.
    cos = torch.cos(angles)  # shape: (max_length, half_dim)
    sin = torch.sin(angles)  # shape: (max_length, half_dim)
    # Limit cosine and sine to the actual sequence length and move to the input device.
    seq_len = x.size(-2)
    cos = cos[:seq_len].to(x.device)
    sin = sin[:seq_len].to(x.device)
    # Reshape for broadcasting to x.
    cos = cos.view(*([1] * (x.ndim - 2)), seq_len, half_dim)
    sin = sin.view(*([1] * (x.ndim - 2)), seq_len, half_dim)
    # Split x into even and odd indexed features.
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    # Apply the rotary transformation.
    x_rotated = torch.cat(
        [x_even * cos - x_odd * sin, x_odd * cos + x_even * sin], dim=-1
    )
    return x_rotated
