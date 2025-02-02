"""
Basic Models (Non-Attention Based)

This module contains baseline sequence models that do not use attention mechanisms:
- FCNModel: A fully connected network that flattens the sequence and processes it via MLP layers.
- CNNModel: A 1D convolutional network with multiple layers.
- LSTMModel: A bidirectional LSTM model.
"""

import torch
import torch.nn as nn
from config import SEQUENCE_LENGTH


class FCNModel(nn.Module):
    """
    Fully Connected Network (FCN) baseline model.

    Flattens the input sequence and processes it through multiple fully connected layers with GELU activations.
    """

    def __init__(self, seq_length: int = SEQUENCE_LENGTH) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length, 256),  # First fully connected layer
            nn.GELU(),
            nn.Linear(256, 256),  # Second fully connected layer
            nn.GELU(),
            nn.Linear(256, seq_length),  # Output layer to reconstruct the sequence
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        out = self.model(x)
        # Reshape output to (batch, 1, seq_length)
        return out.view(batch_size, 1, self.seq_length)


class CNNModel(nn.Module):
    """
    Convolutional Neural Network (CNN) model for sequence tasks.

    Uses several 1D convolution layers with GELU activations to process the input.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),  # First convolution layer
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),  # Second convolution layer
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),  # Third convolution layer
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),  # Fourth convolution layer
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=5, padding=2),  # Output convolution layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LSTMModel(nn.Module):
    """
    LSTM-based model for sequence tasks.

    Uses a bidirectional LSTM to process the input sequence and projects the LSTM output
    back to a single channel.
    """

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        # Project LSTM output back to one channel.
        self.output_proj = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose input from (batch, 1, seq_len) to (batch, seq_len, 1)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        # Project the LSTM output and transpose back to (batch, 1, seq_len)
        return self.output_proj(lstm_out).transpose(1, 2)
