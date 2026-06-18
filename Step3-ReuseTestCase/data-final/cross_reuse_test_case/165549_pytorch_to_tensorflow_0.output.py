import torch
import tensorflow as tf
import numpy as np

def test_multiply_no_nan_shape_preservation():
    """
    Adapted from PyTorch Issue 165549.
    
    The original bug report describes a scenario where `torch.abs` (a unary op) 
    returned a tensor with shape [0] instead of the input shape [4, 4] when 
    running through CPU fallback on a custom device.
    
    This test adapts that logic to `tf.compat.v1.math.multiply_no_nan` (a binary op)
    to verify that the output tensor preserves the correct input shapes and 
    handles the specific NaN/Zero semantics correctly.
    """
    
    # 1. Shape Preservation Test
    # Analogous to: t = torch.randn(4, 4, device='privateuse1')
    # We create inputs with a specific non-zero shape.
    input_shape = (4, 4)
    x = tf.random.normal(input_shape, seed=42)
    y = tf.random.normal(input_shape, seed=43)

    # Perform the operation
    # Analogous to: result = torch.abs(t)
    result = tf.compat.v1.math.multiply_no_nan(x, y)

    # The core assertion from the bug report:
    # Ensure the result is not empty (shape [0]) and matches the input shape.
    assert result.shape == input_shape, (
        f"Shape mismatch detected. Expected {input_shape}, but got {result.shape}. "
        "This mimics the failure mode in the original PyTorch bug where shape was [0]."
    )

    # 2. Semantic Correctness Test
    # Verify the specific behavior of multiply_no_nan:
    # Computes x * y, but if y is 0, the result is 0 (even if x is NaN).
    
    # Case: x is NaN, y is 0 -> Result should be 0
    x_nan = tf.constant([float('nan'), float('nan')])
    y_zero = tf.constant([0.0, 1.0])
    
    result_special = tf.compat.v1.math.multiply_no_nan(x_nan, y_zero)
    
    # Expected: [0.0, nan]
    # 0.0 because nan * 0 -> 0 (per multiply_no_nan spec)
    # nan because nan * 1 -> nan
    expected_values = np.array([0.0, float('nan')])
    
    # Check the first element (0.0)
    np.testing.assert_equal(result_special.numpy()[0], expected_values[0])
    
    # Check the second element (NaN)
    assert np.isnan(result_special.numpy()[1])
    assert np.isnan(expected_values[1])

if __name__ == "__main__":
    test_multiply_no_nan_shape_preservation()
    print("Test passed successfully.")