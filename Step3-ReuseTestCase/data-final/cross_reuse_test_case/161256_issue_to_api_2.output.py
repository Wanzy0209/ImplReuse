import torch
import tensorflow as tf
import numpy as np

def test_float64_matmul_with_dimension_value():
    """
    Test case for float64 matrix multiplication stability, leveraging 
    tf.compat.dimension_value to verify output dimensions.
    
    This test mirrors the PyTorch issue (float64 matmul on large tensors) 
    but translates the logic to TensorFlow to utilize the similar API.
    """
    # Setup: Reproduce the float64 and large tensor conditions from the bug report
    dtype = tf.float64
    rows, cols = 1000, 1000

    # Attempt to use GPU if available to mimic the "device='cuda'" condition
    gpus = tf.config.list_physical_devices('GPU')
    device = '/GPU:0' if gpus else '/CPU:0'
    
    print(f"Running test on device: {device}")

    with tf.device(device):
        # Create random float64 tensors (equivalent to torch.randn)
        x = tf.random.normal((rows, cols), dtype=dtype)
        y = tf.random.normal((rows, cols), dtype=dtype)

        # Perform Matrix Multiplication (equivalent to x @ y)
        z = tf.matmul(x, y)

        # Leverage the Similar API: tf.compat.dimension_value
        # Use the API to extract and verify the dimension values of the result.
        # This ensures the operation completed and the shape is as expected.
        dim_0 = tf.compat.dimension_value(z.shape[0])
        dim_1 = tf.compat.dimension_value(z.shape[1])

        # Assertions
        assert dim_0 == rows, f"Expected dimension 0 to be {rows}, got {dim_0}"
        assert dim_1 == cols, f"Expected dimension 1 to be {cols}, got {dim_1}"

        # Access a specific value to ensure no segfault occurs on memory access
        # (mirroring the print(z[0, 0]) in the original bug report)
        val = z[0, 0]
        
        # Verify the value is finite (not NaN or Inf)
        assert tf.math.is_finite(val), f"Result value at [0,0] is not finite: {val}"

        print("Test passed successfully.")

if __name__ == "__main__":
    test_float64_matmul_with_dimension_value()