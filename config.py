"""
Configuration Parameters

This module contains all the configuration parameters used across the project.
Centralizing these parameters makes it easier to modify the behavior of the entire system.
"""

from utils.environment import get_device

# Device Setup
DEVICE: str = get_device()  # Automatically determine the computational device

# Training Parameters
BATCH_SIZE: int = 128
NUM_TRAIN_SAMPLES: int = BATCH_SIZE * 100  # Number of samples per epoch
NUM_EPOCHS: int = 50
LEARNING_RATE: float = 0.0003

# Data Generation Parameters
SEQUENCE_LENGTH: int = 400  # Length of the synthetic sequence
MIN_WIDTH: int = 10  # Minimum width of a shape
MAX_WIDTH: int = 40  # Maximum width of a shape
AMPLITUDE_MIN: int = 10  # Minimum amplitude (absolute value)
AMPLITUDE_MAX: int = 50  # Maximum amplitude (absolute value)
NOISE_STD: float = 3.0  # Standard deviation for added noise

# 2D Shape Generation Parameters
IMAGE_HEIGHT: int = 128
IMAGE_WIDTH: int = 128
SHAPE_MIN_RADIUS: int = 10  # Minimum radius for circles
SHAPE_MAX_RADIUS: int = 20  # Maximum radius for circles
SHAPE_MIN_RECT_SIZE: int = 10  # Minimum width/height for rectangles
SHAPE_MAX_RECT_SIZE: int = 30  # Maximum width/height for rectangles
SHAPE_MIN_TRI_SIZE: int = 20  # Minimum side length for triangles
SHAPE_MAX_TRI_SIZE: int = 30  # Maximum side length for triangles
SHAPE_HOLLOW_MIN_THICKNESS: int = 1  # Minimum line thickness for hollow shapes
SHAPE_HOLLOW_MAX_THICKNESS: int = 3  # Maximum line thickness for hollow shapes
SHAPE_COLOR_VARIATION: int = 64  # Maximum +/- variation in RGB values
SHAPE_NOISE_STD: int = 128  # Standard deviation for noise in 2D shapes
SHAPE_NOISE_SIGMA: float = 1.0  # Smoothing factor for 2D shape noise

# Model Parameters
NUM_HEADS: int = 4  # Number of attention heads
HIDDEN_DIM: int = 64  # Hidden dimension for models
KEY_DIM: int = 64  # Key dimension for attention
DROPOUT_RATE: float = 0.1  # Dropout rate for attention layers
STOCHASTIC_DEPTH_PROB: float = (
    0.1  # Probability for stochastic depth in EnhancedAttentionModel
)
LAYER_SCALE_INIT_VALUE: float = 1e-5  # Initial value for layer scaling

# Positional Encoding Parameters
PE_MIN_FREQ: float = 1e-4  # Minimum frequency for positional encoding
PE_MAX_LENGTH: int = 1000  # Maximum sequence length for positional encoding
ROPE_BASE: int = 10000  # Base for rotary positional encoding

# Visualization Parameters
DPI: int = 300  # DPI for saved figures
VALIDATION_PLOT_SAMPLES: int = 5  # Number of validation samples to plot
COLORMAP: str = "Spectral_r"  # Default colormap for plots

# Optimizer Defaults
DEFAULT_OPTIMIZER: str = "adam"  # Options: 'adam', 'adamw', 'rmsprop'
DEFAULT_WEIGHT_DECAY: float = 0.0
DEFAULT_MOMENTUM: float = 0.9

# Scheduler Defaults
DEFAULT_SCHEDULER: str = "cosine"  # Options: 'cosine', 'cosine_with_restarts'
DEFAULT_ETA_MIN: float = 0.0  # Minimum learning rate
DEFAULT_T_MAX: int = 50  # Maximum iterations for cosine annealing
DEFAULT_T_0: int = 50  # Iterations for first restart (for cosine_with_restarts)
DEFAULT_T_MULT: int = 1  # Multiplier for cosine_with_restarts
DEFAULT_WARMUP_EPOCHS: int = 3  # Epochs for warmup phase

# Loss Parameters
DEFAULT_LOSS: str = "mse"  # Options: 'mse', 'mae', 'huber'
HUBER_DELTA: float = 1.0  # Delta value for Huber loss if used

# Vision Transformer Parameters
VISION_TRANSFORMER_IMAGE_SIZE: tuple = (128, 128)  # (height, width)
VISION_TRANSFORMER_PATCH_SIZE: int = 16  # Size of each square patch
VISION_TRANSFORMER_EMBED_DIM: int = 256  # Embedding dimension for patch tokens
VISION_TRANSFORMER_ENCODER_DEPTH: int = 4  # Number of transformer encoder blocks
VISION_TRANSFORMER_DECODER_DEPTH: int = 4  # Number of transformer decoder blocks
VISION_TRANSFORMER_NUM_HEADS: int = 8  # Number of attention heads
VISION_TRANSFORMER_DROPOUT: float = 0.1  # Dropout probability for transformer layers
