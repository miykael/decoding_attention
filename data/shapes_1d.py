"""
Refactored module for generating and visualizing synthetic 1D shape data.

This module provides functions to generate synthetic sequences that contain non-overlapping shapes,
and a PyTorch Dataset to utilize these generated sequences. Shapes can be drawn as triangles,
rectangles, or semicircles with randomness in position, width, and amplitude.
The input sequence is further corrupted with smooth noise.
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import configuration parameters
from config import (
    SEQUENCE_LENGTH,
    MIN_WIDTH,
    MAX_WIDTH,
    AMPLITUDE_MIN,
    AMPLITUDE_MAX,
    NOISE_STD,
)

# Allowed shape types; each string encodes both sign and shape
ALLOWED_SHAPES = [
    "pos_triangle",
    "pos_rectangle",
    "pos_semicircle",
    "neg_triangle",
    "neg_rectangle",
    "neg_semicircle",
]


def generate_smooth_noise(
    length: int, noise_std: float, sigma: float = 5
) -> np.ndarray:
    """
    Generate smooth Gaussian noise by convolving white noise with a Gaussian kernel.

    Args:
        length: The length of the noise sequence.
        noise_std: Standard deviation of the white noise.
        sigma: Standard deviation for the Gaussian kernel smoothing (default is 5).

    Returns:
        A NumPy array of smooth noise of shape (length,).
    """
    # Generate white noise
    white_noise: np.ndarray = np.random.normal(0, noise_std, length)
    # Determine kernel size to cover roughly +/- 3 sigma
    kernel_size: int = int(6 * sigma) + 1
    x: np.ndarray = np.arange(kernel_size) - kernel_size // 2
    # Create Gaussian kernel and normalize it
    kernel: np.ndarray = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    # Convolve white noise with Gaussian kernel
    smooth_noise: np.ndarray = np.convolve(white_noise, kernel, mode="same")
    return smooth_noise


def draw_triangle(seq: np.ndarray, start: int, width: int, amplitude: float) -> None:
    """
    Draw a triangle shape into the sequence array.

    This function draws an upward triangle by linearly increasing the values to a peak and then decreasing.

    Args:
        seq: NumPy array representing the sequence.
        start: Starting index to draw the triangle.
        width: Width of the triangle.
        amplitude: Peak amplitude of the triangle.
    """
    end: int = start + width
    peak: int = start + width // 2
    # Draw rising slope from start to peak
    if peak > start:
        seq[start:peak] = np.linspace(0, amplitude, num=peak - start, endpoint=False)
    # Draw falling slope from peak to end
    if end > peak:
        seq[peak:end] = np.linspace(amplitude, 0, num=end - peak, endpoint=True)


def draw_rectangle(seq: np.ndarray, start: int, width: int, amplitude: float) -> None:
    """
    Draw a rectangle shape into the sequence array.

    Fills a segment of the sequence with a constant amplitude.

    Args:
        seq: NumPy array representing the sequence.
        start: Starting index to draw the rectangle.
        width: Width of the rectangle.
        amplitude: Constant amplitude value for the rectangle.
    """
    end: int = start + width
    seq[start:end] = amplitude


def draw_semicircle(seq: np.ndarray, start: int, width: int, amplitude: float) -> None:
    """
    Draw a semicircular shape into the sequence array.

    Calculates semicircular values based on the distance from the center.

    Args:
        seq: NumPy array representing the sequence.
        start: Starting index to draw the semicircle.
        width: Width of the semicircle.
        amplitude: Amplitude scalar for the semicircle.
    """
    end: int = start + width
    indices: np.ndarray = np.arange(start, end)
    if width > 1:
        # Normalize indices to range [-1, 1]
        x: np.ndarray = (2 * (indices - start) / (width - 1)) - 1
    else:
        x = np.zeros_like(indices, dtype=float)
    # Compute the semicircular shape and clip negative values
    semicircle: np.ndarray = amplitude * np.sqrt(np.clip(1 - x**2, 0, 1))
    seq[start:end] = semicircle


def draw_shape(
    seq: np.ndarray, shape_type: str, start: int, width: int, amplitude: float
) -> None:
    """
    Draw a shape into the sequence array based on the provided shape type.

    Args:
        seq: NumPy array representing the sequence.
        shape_type: The type of the shape ('triangle', 'rectangle', or 'semicircle').
        start: Starting index for drawing the shape.
        width: Width (or duration) of the shape.
        amplitude: Amplitude to use for drawing the shape.

    Raises:
        ValueError: If the shape type is unknown.
    """
    if "triangle" in shape_type:
        draw_triangle(seq, start, width, amplitude)
    elif "rectangle" in shape_type:
        draw_rectangle(seq, start, width, amplitude)
    elif "semicircle" in shape_type:
        draw_semicircle(seq, start, width, amplitude)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")


def generate_shapes(
    size: int = SEQUENCE_LENGTH,
    min_width: int = MIN_WIDTH,
    max_width: int = MAX_WIDTH,
    amplitude_min: float = AMPLITUDE_MIN,
    amplitude_max: float = AMPLITUDE_MAX,
    noise_std: float = NOISE_STD,
) -> tuple:
    """
    Generate a synthetic sequence with non-overlapping shapes.

    The function selects 3 to 4 distinct shape types from ALLOWED_SHAPES and, for each type,
    generates 1 to 4 occurrences. The input sequence is drawn using each occurrence's amplitude
    and then has smooth noise added; the target sequence is generated using the group-average
    amplitude for each shape type.

    Args:
        size: Length of the sequence (default from SEQUENCE_LENGTH).
        min_width: Minimum width of a shape (default from MIN_WIDTH).
        max_width: Maximum width of a shape (default from MAX_WIDTH).
        amplitude_min: Minimum amplitude for shapes (default from AMPLITUDE_MIN).
        amplitude_max: Maximum amplitude for shapes (default from AMPLITUDE_MAX).
        noise_std: Standard deviation for added noise (default from NOISE_STD).

    Returns:
        A tuple (input_seq, target_seq) where each is a NumPy array of shape (size,).
    """
    input_seq: np.ndarray = np.zeros(size)
    target_seq: np.ndarray = np.zeros(size)

    # Randomly select 3 or 4 distinct shape types
    num_shapes: int = random.randint(3, 4)
    shape_types = random.sample(ALLOWED_SHAPES, num_shapes)

    shape_logs: list = []  # To log shape properties (type, position, width, amplitude)
    placed_intervals: list = []  # To keep track of used intervals (avoid overlap)

    def is_overlapping(start: int, end: int, intervals: list) -> bool:
        """
        Check if the interval [start, end) overlaps with any intervals in a list.

        Args:
            start: Starting index of the new interval.
            end: Ending index of the new interval.
            intervals: List of existing intervals as tuples (s, e).

        Returns:
            True if there is any overlap; otherwise, False.
        """
        for s, e in intervals:
            if not (end <= s or start >= e):
                return True
        return False

    # For each selected shape type, determine its occurrences
    for shape_type in shape_types:
        num_occurrences: int = random.randint(1, 4)
        for _ in range(num_occurrences):
            placed: bool = False
            attempts: int = 0
            while not placed and attempts < 100:
                width: int = random.randint(min_width, max_width)
                start: int = random.randint(0, size - width)
                end: int = start + width
                if not is_overlapping(start, end, placed_intervals):
                    placed_intervals.append((start, end))
                    amplitude: float = random.uniform(amplitude_min, amplitude_max)
                    # Invert amplitude if the shape type denotes a negative shape
                    if "neg" in shape_type:
                        amplitude = -amplitude
                    shape_logs.append(
                        {
                            "type": shape_type,
                            "pos": start,
                            "width": width,
                            "amplitude": amplitude,
                        }
                    )
                    placed = True
                attempts += 1

    # Draw each shape in the input sequence with its own amplitude
    for log in shape_logs:
        draw_shape(input_seq, log["type"], log["pos"], log["width"], log["amplitude"])
    # Add smooth noise to the input sequence
    input_seq += generate_smooth_noise(size, noise_std)

    # Compute group-average amplitude for each shape type
    group_averages: dict = {}
    for log in shape_logs:
        stype: str = log["type"]
        group_averages.setdefault(stype, []).append(log["amplitude"])
    for stype in group_averages:
        group_averages[stype] = np.mean(group_averages[stype])

    # Generate the target sequence using the group-average amplitude for each shape type
    for log in shape_logs:
        avg_amp: float = group_averages[log["type"]]
        draw_shape(target_seq, log["type"], log["pos"], log["width"], avg_amp)

    return input_seq, target_seq


class ShapeDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for 1D shape sequences.

    Each sample is generated on the fly using generate_shapes and returned as a pair:
    - The input sequence (with noise) as a FloatTensor with an added channel dimension.
    - The target sequence (using group-average amplitudes) as a FloatTensor with an added channel dimension.
    """

    def __init__(self, num_samples: int = 10000, **kwargs) -> None:
        """
        Initialize the ShapeDataset.

        Args:
            num_samples: Number of samples to generate.
            kwargs: Additional keyword arguments to pass to generate_shapes.
        """
        self.num_samples: int = num_samples
        self.kwargs: dict = kwargs

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        """
        Generate and return a sample from the dataset.

        Args:
            idx: Index of the sample (not used since data is generated on the fly).

        Returns:
            A tuple of two FloatTensors:
            - Input sequence with shape [1, size].
            - Target sequence with shape [1, size].
        """
        inp, target = generate_shapes(**self.kwargs)
        return (
            torch.FloatTensor(inp).unsqueeze(0),
            torch.FloatTensor(target).unsqueeze(0),
        )


def visualize_sample() -> None:
    """
    Generate a sample and visualize the input (with noise) and target (averaged shapes) sequences.

    This function is useful for demonstration and debugging.
    """
    inp, target = generate_shapes()
    plt.figure(figsize=(12, 6))
    plt.plot(inp, label="Input Sequence (with noise)", color="blue", alpha=0.7)
    plt.plot(
        target,
        label="Target Sequence (averaged shapes)",
        color="orange",
        linestyle="--",
        linewidth=2,
    )
    plt.title("Input vs Target Sequence")
    plt.xlabel("Position")
    plt.ylabel("Amplitude")
    plt.ylim(-60, 60)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_sample()
