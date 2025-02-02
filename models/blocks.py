"""
Re-usable Building Blocks for Attention and Related Layers

This module contains core components that implement various attention mechanisms,
including basic self-attention, multi-head self-attention, and efficient attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoding import rotary_positional_encoding as rotary_positional_encoding


class SelfAttentionLayer(nn.Module):
    """
    Basic self-attention layer with optional flags to return attention weights and Q, K, V matrices.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        key_dim: int,
        use_rope: bool = False,
        max_length: int = 1000,
    ) -> None:
        super().__init__()
        self.use_rope = use_rope
        self.key_dim = key_dim
        # 1x1 convolutional layers for the queries, keys, and values.
        self.query = nn.Conv1d(in_dim, key_dim, kernel_size=1)
        self.key = nn.Conv1d(in_dim, key_dim, kernel_size=1)
        self.value = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        if use_rope:
            # Assign the RoPE function for later use.
            self.rope = rotary_positional_encoding

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_qkv: bool = False,
    ):
        """
        Forward pass of the self-attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, seq_len).
            return_attention (bool): If True, returns the attention weights with the output.
            return_qkv (bool): If True, returns the Q, K and V matrices (along with attention).

        Returns:
            - If both flags are False:
                  torch.Tensor: The attended output.
            - If return_attention is True:
                  (output, attention_weights)
            - If return_qkv is True:
                  (output, attention_weights, (q, k, v))
        """
        batch_size, _, seq_len = x.size()
        queries = self.query(x)  # (batch, key_dim, seq_len)
        keys_ = self.key(x)  # (batch, key_dim, seq_len)
        values = self.value(x)  # (batch, out_dim, seq_len)

        if self.use_rope:
            queries = queries.transpose(1, 2)  # (batch, seq_len, key_dim)
            keys_ = keys_.transpose(1, 2)
            # Apply RoPE to queries and keys.
            queries = self.rope(queries, self.key_dim, max_length=seq_len)
            keys_ = self.rope(keys_, self.key_dim, max_length=seq_len)
            queries = queries.transpose(1, 2)  # back to (batch, key_dim, seq_len)
            keys_ = keys_.transpose(1, 2)

        # Prepare Q, K, V matrices.
        q = queries.transpose(1, 2)  # (batch, seq_len, key_dim)
        k = keys_.view(batch_size, self.key_dim, seq_len)  # (batch, key_dim, seq_len)
        v = values.transpose(1, 2)  # (batch, seq_len, out_dim)

        # Scaled dot-product attention.
        scores = torch.bmm(q, k) / math.sqrt(self.key_dim)  # (batch, seq_len, seq_len)
        attention = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        out = torch.bmm(attention, v)  # (batch, seq_len, out_dim)
        out = out.transpose(1, 2).contiguous()  # reshape to (batch, out_dim, seq_len)

        if return_qkv:
            return out, attention, (q, k, v)
        elif return_attention:
            return out, attention
        else:
            return out


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention layer with optional output of attention weights and Q, K, V matrices.
    """

    def __init__(self, in_dim: int, out_dim: int, key_dim: int, n_heads: int) -> None:
        super().__init__()
        assert key_dim % n_heads == 0, "key_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = key_dim // n_heads
        self.query = nn.Conv1d(in_dim, key_dim, kernel_size=1)
        self.key = nn.Conv1d(in_dim, key_dim, kernel_size=1)
        self.value = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        self.output_proj = nn.Conv1d(out_dim, out_dim, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_qkv: bool = False,
    ):
        """
        Forward pass for multi-head self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, seq_len).
            return_attention (bool): If True, return the attention weights.
            return_qkv (bool): If True, return the query, key, and value matrices.

        Returns:
            - If both flags are False:
                  torch.Tensor: The attended output.
            - If return_attention is True:
                  (output, attention_weights)
            - If return_qkv is True:
                  (output, attention_weights, (q, k, v))
        """
        batch_size, _, seq_len = x.size()
        # Compute queries, keys, and values.
        queries = self.query(x).view(batch_size, self.n_heads, self.head_dim, seq_len)
        keys_ = self.key(x).view(batch_size, self.n_heads, self.head_dim, seq_len)
        values = self.value(x).view(batch_size, self.n_heads, -1, seq_len)

        # For attention calculation, transpose queries.
        q = queries.transpose(2, 3)  # (batch, n_heads, seq_len, head_dim)
        scores = torch.matmul(q, keys_) / math.sqrt(
            self.head_dim
        )  # (batch, n_heads, seq_len, seq_len)
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(
            attention, values.transpose(2, 3)
        )  # (batch, n_heads, seq_len, out_per_head)
        out = out.transpose(2, 3).contiguous().view(batch_size, -1, seq_len)
        out = self.output_proj(out)

        if return_qkv:
            return out, attention, (q, keys_, values)
        elif return_attention:
            return out, attention
        else:
            return out


class EfficientAttention(nn.Module):
    """
    Memory-efficient attention.

    Uses linear layers for projections and supports optional causal masking. Can
    also integrate Rotary Positional Encoding (RoPE) if enabled.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = False,
        use_rope: bool = False,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # Ensure that the feature dimension is divisible by the number of heads.
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5  # Scaling factor for attention scores
        self.causal = causal
        self.use_rope = use_rope
        # Project input from shape (batch, seq_len, 1) to (batch, seq_len, dim)
        self.input_proj = nn.Linear(1, dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        if use_rope:
            self.rope = rotary_positional_encoding
        self.final_proj = nn.Linear(dim, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x shape: (batch, seq_len, 1) is projected to (batch, seq_len, dim)
        x = self.input_proj(x)
        batch_size, seq_len, _ = x.shape
        # Compute query, key, and value tensors and reshape for multi-head attention.
        q = self.to_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.to_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.to_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        if self.use_rope:
            # Apply RoPE by flattening the head dimensions and then reshaping.
            q = self.rope(
                q.view(batch_size, seq_len, -1), self.dim, max_length=seq_len
            ).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = self.rope(
                k.view(batch_size, seq_len, -1), self.dim, max_length=seq_len
            ).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Rearrange tensors for attention calculation.
        q = q.transpose(1, 2)  # shape: (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        if self.causal:
            # Create and apply a causal mask.
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        out = torch.matmul(attention_weights, v)
        # Merge heads and project.
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.to_out(out)
        out = self.final_proj(out)
        # Permute output to shape (batch, 1, seq_len)
        return out.permute(0, 2, 1)


class EfficientAttentionWithRoPE(EfficientAttention):
    """
    Efficient Attention variant with integrated Rotary Positional Encoding (RoPE).

    Simply reinitializes the RoPE component to use the full feature dimension.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = False,
        max_length: int = 512,
    ) -> None:
        super().__init__(
            dim, num_heads, dropout, causal, use_rope=True, max_length=max_length
        )
        self.rope = rotary_positional_encoding

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple:
        batch_size, seq_len, _ = x.shape
        # Compute queries, keys, and values.
        q = self.to_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.to_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.to_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Apply RoPE transformation.
        q = self.rope(
            q.view(batch_size, seq_len, -1), self.dim, max_length=seq_len
        ).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.rope(
            k.view(batch_size, seq_len, -1), self.dim, max_length=seq_len
        ).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.to_out(out)
        return out, attention_weights
