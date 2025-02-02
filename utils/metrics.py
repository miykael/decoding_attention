import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in the given model.

    Args:
        model: The PyTorch model to analyze.

    Returns:
        The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lstm_flop_jit(inputs: list, outputs: list) -> int:
    """
    Calculate FLOPs for an LSTM layer.

    This handler is used to register with fvcore for LSTM operations because
    fvcore does not compute LSTM FLOPs correctly out of the box.

    Args:
        inputs: A list containing the input tensor (batch_size, seq_len, input_size).
        outputs: A list containing the output tensor (batch_size, seq_len, hidden_size).

    Returns:
        The total number of floating point operations for the LSTM layer.
    """

    # Helper function to extract size information.
    def get_size(t):
        try:
            # If t is a regular tensor.
            return list(t.size())
        except AttributeError:
            # If t is a traced value, get sizes from its type.
            return list(t.type().sizes())

    x = inputs[0]
    input_shape = get_size(x)  # Expected: [batch_size, seq_len, input_size]
    y = outputs[0]
    output_shape = get_size(y)  # Expected: [batch_size, seq_len, hidden_size]

    batch_size, seq_len, input_size = input_shape
    hidden_size = output_shape[-1]

    # Determine bidirectionality: if hidden_size equals 2 * input_size, assume bidirectional.
    bidirectional_factor = 2 if hidden_size == 2 * input_size else 1
    hidden_size = hidden_size // bidirectional_factor

    # Calculate operations for gates: input, forget, cell, and output (4 gates)
    gate_ops = 4 * (
        input_size * hidden_size  # Input-to-hidden weights
        + hidden_size * hidden_size  # Hidden-to-hidden weights
        + hidden_size  # Biases
        + hidden_size * 2  # Extra operations for gate activations
    )

    # Additional operations for cell state (e.g. tanh, sigmoid, and elementwise ops)
    cell_ops = 5 * hidden_size

    # Total operations per time step, scaled by sequence length, batch size, and bidirectionality.
    ops_per_timestep = gate_ops + cell_ops
    total_flops = ops_per_timestep * seq_len * batch_size * bidirectional_factor

    return total_flops


def calculate_flops_for_rnn(flops: FlopCountAnalysis) -> FlopCountAnalysis:
    """
    Register the custom LSTM handler with the FLOPs counter.

    Args:
        flops: An instance of FlopCountAnalysis.

    Returns:
        Updated FlopCountAnalysis with the LSTM handler registered.
    """
    flops.set_op_handle("aten::lstm", lstm_flop_jit)
    return flops


def default_flop_handler(inputs: list, outputs: list) -> int:
    """
    Default handler for operations not explicitly supported.

    For instance, we ignore FLOPs for GELU since on MPS this operator isn't supported.
    """
    return 0


def count_flops(model: nn.Module, input: torch.Tensor) -> float:
    """
    Count FLOPs for the given model using a dummy input.

    This function wraps fvcore's FlopCountAnalysis utility and
    registers custom handlers for LSTM and GELU if needed.

    Args:
        model: The PyTorch model to analyze.
        input: A dummy input tensor or a tuple of input tensors.

    Returns:
        The total number of floating point operations (FLOPs).
    """
    # Ensure input is provided as a tuple
    input_tuple = input if isinstance(input, tuple) else (input,)
    flops = FlopCountAnalysis(model, input_tuple)
    flops = calculate_flops_for_rnn(flops)
    # Register a default handler to avoid unsupported operator errors (e.g., GELU)
    flops.set_op_handle("aten::gelu", default_flop_handler)
    return flops.total()
