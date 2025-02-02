"""Environment setup utilities for PyTorch models.

This module provides functions for ensuring reproducibility and optimal device selection
across various hardware configurations.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all random number generators.

    Args:
        seed: Integer seed value for random number generation.

    Note:
        This sets seeds for Python's random module, NumPy, and PyTorch (both CPU and GPU).
        Deterministic behavior is enforced in cuDNN, at the potential cost of performance.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy's random generator
    np.random.seed(seed)
    # Set seed for PyTorch CPU operations
    torch.manual_seed(seed)
    # If CUDA is available, set seeds for GPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Current GPU
        torch.cuda.manual_seed_all(seed)  # All GPUs
        torch.backends.cudnn.deterministic = (
            True  # Use deterministic algorithms in cuDNN
        )
        torch.backends.cudnn.benchmark = False  # Disable benchmark for reproducibility


def get_device() -> str:
    """
    Determine the optimal available compute device.

    Returns:
        A string: 'cuda' if an NVIDIA GPU is available, 'mps' for Apple M-series GPUs, otherwise 'cpu'.

    Note:
        The function prioritizes CUDA, then MPS, and defaults to CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    # Check for Apple's MPS support
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
