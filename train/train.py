"""
Training Utilities for Attention Models

This module provides functions for training, validating, and evaluating models. In addition,
it includes utilities to extract attention patterns for visualization. All key steps are documented
with inline comments.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.shapes_1d import ShapeDataset
from config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NUM_TRAIN_SAMPLES,
    SEQUENCE_LENGTH,
    NOISE_STD,
    DEFAULT_LOSS,
    HUBER_DELTA,
)
from train.scheduler import get_scheduler
from train.optimizer import get_optimizer


def get_criterion(loss_type: str = DEFAULT_LOSS) -> nn.Module:
    """
    Get the loss criterion based on the specified type.

    Args:
        loss_type: The type of loss to use ('mse', 'mae', or 'huber').

    Returns:
        A PyTorch loss module.
    """
    loss_type = loss_type.lower()
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.HuberLoss(delta=HUBER_DELTA)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def train_model(
    model: nn.Module,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
    epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    loss_type: str = DEFAULT_LOSS,
) -> tuple:
    """
    Train the provided model on the synthetic ShapeDataset.

    Args:
        model: The PyTorch model to be trained.
        device: The device to run training on.
        batch_size: The batch size for training.
        epochs: The number of training epochs.
        learning_rate: The learning rate for the optimizer.
        loss_type: The type of loss function to use.

    Returns:
        A tuple containing:
          - losses: A list of average epoch losses.
          - training_time: The total training time in seconds.
    """
    # Create the training dataset and loader
    dataset = ShapeDataset(
        num_samples=NUM_TRAIN_SAMPLES, size=SEQUENCE_LENGTH, noise_std=NOISE_STD
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    model.to(device)

    # Setup optimizer and scheduler
    optimizer = get_optimizer(model.parameters(), learning_rate=learning_rate)
    scheduler = get_scheduler(optimizer)

    # Get the specified loss criterion
    criterion = get_criterion(loss_type)
    losses = []

    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            input_seq, target_seq = [b.to(device) for b in batch]
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Step the scheduler at the end of each epoch
        scheduler.step()

        # Log the current learning rate
        current_lr = scheduler.get_last_lr()[0]

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

    training_time = time.time() - start_time
    print(
        f"Training completed in {training_time:.2f} seconds. {training_time / epochs:.2f} seconds per epoch"
    )
    return losses, training_time / epochs


def evaluate_model(model: nn.Module, input_seq, device: str = DEVICE) -> tuple:
    """
    Evaluate the model on a single input sequence.

    Args:
        model: The PyTorch model to evaluate.
        input_seq: A sequence (list or array) to evaluate.
        device: The device used for running inference.

    Returns:
        A tuple containing:
          - prediction: The model's output for the input sequence.
          - inference_time: Time (in seconds) taken for inference.
    """
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        # Prepare the input: add batch and channel dimensions.
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(0).to(device)
        output = model(input_tensor)
    inference_time = time.time() - start_time
    # Remove batch and channel dimensions before returning.
    return output[0, 0].cpu(), inference_time
