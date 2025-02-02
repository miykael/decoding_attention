"""
Loss Functions for Training

This module provides functions to retrieve loss functions by name and to set up loss
functions from a configuration. Note that while the setup_loss_functions() function
allows multiple losses per feature, many training procedures may only need a single loss.
"""

import torch.nn as nn
from config import HUBER_DELTA


def get_loss_function(loss_name: str) -> nn.Module:
    """
    Retrieve a loss function instance based on its name.

    Args:
        loss_name: A string representing the loss function name.
                   Options: 'mse', 'mae', or 'huber'.

    Returns:
        A PyTorch loss function (nn.Module).

    Raises:
        ValueError: If the provided loss name is not supported.
    """
    # Dictionary mapping supported loss names to their corresponding PyTorch loss modules.
    loss_functions = {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "huber": nn.HuberLoss(delta=HUBER_DELTA),  # Use HUBER_DELTA from config
    }
    if loss_name.lower() not in loss_functions:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    return loss_functions[loss_name.lower()]


def setup_loss_functions(cfg: dict) -> dict:
    """
    Setup loss functions from configuration with weights for each feature.

    The configuration is expected to have a field named 'loss' where each key corresponds
    to a feature and its value is a list of loss configurations. Each loss configuration
    must have attributes 'name' and 'weight'.

    Args:
        cfg: A dictionary containing a 'loss' key mapping feature names to a list of loss configs.

    Returns:
        A dictionary mapping feature names to a list of (loss_function, weight) tuples.

    Note:
        Some training setups may only require a single loss function. If that is the case,
        this helper function can be simplified.
    """
    loss_functions = {}

    # Iterate over each feature's loss configuration.
    for feature, losses in cfg.get("loss", {}).items():
        # Skip if no losses are defined for the feature.
        if not losses:
            continue

        feature_losses = []
        # Process each loss configuration.
        for loss_cfg in losses:
            # Retrieve the loss function based on the provided loss name.
            loss_fn = get_loss_function(loss_cfg.name)
            # Append the loss function along with its weight.
            feature_losses.append((loss_fn, loss_cfg.weight))
        loss_functions[feature] = feature_losses

    return loss_functions
