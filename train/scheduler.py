"""
Scheduler Utilities for Training

This module provides functions to create learning rate schedulers with optional linear warmup.
Supported schedulers include CosineAnnealingLR and CosineAnnealingWarmRestarts.
"""

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    _LRScheduler,
)
from torch.optim import Optimizer
from config import (
    DEFAULT_SCHEDULER,
    DEFAULT_ETA_MIN,
    DEFAULT_T_MAX,
    DEFAULT_T_0,
    DEFAULT_T_MULT,
    DEFAULT_WARMUP_EPOCHS,
)


class WarmupSchedulerWrapper(_LRScheduler):
    """
    Wraps a scheduler to add linear warmup for the first few epochs.

    During warmup, the learning rate increases linearly from eta_min to the initial lr.
    After warmup, the wrapped scheduler takes over.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        after_scheduler: _LRScheduler,
        eta_min: float = 0.0,
    ):
        """
        Initialize the warmup wrapper.

        Args:
            optimizer: The optimizer whose learning rate is being scheduled.
            warmup_epochs: Number of epochs for linear warmup.
            after_scheduler: The scheduler to use after warmup.
            eta_min: Minimum learning rate to start warmup from.
        """
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.eta_min = eta_min
        self.finished = False

        # Get the initial learning rate from the optimizer
        self.initial_lr = optimizer.param_groups[0]["lr"]

        # Initialize parent class
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Calculate the learning rate based on the current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.eta_min + alpha * (self.initial_lr - self.eta_min)
                for _ in self.base_lrs
            ]
        # After warmup, use the wrapped scheduler
        self.after_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
        return self.after_scheduler.get_last_lr()


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = DEFAULT_SCHEDULER,
    eta_min: float = DEFAULT_ETA_MIN,
    T_max: int = DEFAULT_T_MAX,
    T_0: int = DEFAULT_T_0,
    T_mult: int = DEFAULT_T_MULT,
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
) -> _LRScheduler:
    """
    Create a learning rate scheduler with optional warmup for the given optimizer.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: Type of scheduler ('cosine' or 'cosine_with_restarts').
        eta_min: Minimum learning rate.
        T_max: Maximum number of iterations for cosine annealing.
        T_0: Number of iterations for the first restart.
        T_mult: Factor by which T_0 is multiplied after each restart.
        warmup_epochs: Number of epochs for linear warmup.

    Returns:
        A PyTorch learning rate scheduler instance, wrapped with warmup if warmup_epochs > 0.

    Raises:
        ValueError: If the scheduler type is not supported or required parameters are missing.
    """
    scheduler_type = scheduler_type.lower()

    # Create the main scheduler
    if scheduler_type == "cosine":
        if T_max is None:
            raise ValueError("T_max must be provided for cosine scheduler.")
        main_scheduler = CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=T_max)
    elif scheduler_type == "cosine_with_restarts":
        if T_0 is None:
            raise ValueError("T_0 must be provided for cosine_with_restarts scheduler.")
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer, eta_min=eta_min, T_0=T_0, T_mult=T_mult
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    # Wrap with warmup if warmup_epochs > 0
    if warmup_epochs > 0:
        return WarmupSchedulerWrapper(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            after_scheduler=main_scheduler,
            eta_min=eta_min,
        )

    return main_scheduler
