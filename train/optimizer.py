"""
Optimizer Utilities for Training

This module provides functions to create PyTorch optimizers based on explicitly provided
parameters rather than relying on a generic configuration. Supported optimizers include
Adam, AdamW, and RMSprop.
"""

from torch.optim import Adam, AdamW, RMSprop, Optimizer
from config import (
    LEARNING_RATE,
    DEFAULT_OPTIMIZER,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_MOMENTUM,
)


def get_optimizer(
    model_params,
    optimizer_type: str = DEFAULT_OPTIMIZER,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    momentum: float = DEFAULT_MOMENTUM,
) -> Optimizer:
    """
    Create an optimizer instance for the given model parameters.

    Args:
        model_params: The parameters of the model to be optimized.
        optimizer_type: A string indicating the type of optimizer ('adam', 'adamw', or 'rmsprop').
                        Defaults to DEFAULT_OPTIMIZER from config.
        learning_rate: The learning rate for the optimizer. Defaults to LEARNING_RATE.
        weight_decay: Weight decay (L2 penalty) value; defaults to DEFAULT_WEIGHT_DECAY.
        momentum: Momentum factor (only applicable for RMSprop). Defaults to DEFAULT_MOMENTUM.

    Returns:
        A PyTorch Optimizer instance configured with the specified settings.

    Raises:
        ValueError: If the specified optimizer type is not supported.
    """
    optimizer_type = optimizer_type.lower()
    # Map supported optimizer types to the corresponding PyTorch optimizer class.
    optimizer_class = {"adam": Adam, "adamw": AdamW, "rmsprop": RMSprop}.get(
        optimizer_type
    )
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Build keyword arguments for the optimizer.
    optimizer_kwargs = {"lr": learning_rate}

    # Set momentum for RMSprop optimizer.
    if optimizer_type == "rmsprop":
        optimizer_kwargs["momentum"] = momentum

    # Include weight decay if specified.
    if weight_decay:
        optimizer_kwargs["weight_decay"] = weight_decay

    return optimizer_class(model_params, **optimizer_kwargs)
