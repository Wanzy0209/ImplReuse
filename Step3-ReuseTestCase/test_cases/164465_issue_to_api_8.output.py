import torch
import tensorflow as tf
import numpy as np

# The original issue (164465) involves a crash in PyTorch's inductor when handling
# torch.int64 types with specific operations (iota/arange + max).
# The similar API, tf.keras.ops.true_divide, contains specific logic to handle
# int64 inputs by casting them to float32 to avoid float64 issues (see _avoid_float64).
# This test verifies that tf.keras.ops.true_divide correctly handles int64 inputs
# without crashing and produces the expected float output.

def test_true_divide_int64_handling():
    # Create int64 inputs, mirroring the dtype focus of the original bug report
    # where dtype=torch.int64 was a key factor.
    x1 = tf.constant([10, 20, 30], dtype=tf.int64)
    x2 = tf.constant([2, 5, 6], dtype=tf.int64)

    # Call the similar API
    result = tf.keras.ops.true_divide(x1, x2)

    # Expected result of the division
    expected = np.array([5.0, 4.0, 5.0])

    # Verify the output values are correct
    assert np.allclose(result.numpy(), expected), f"Values mismatch: {result.numpy()} vs {expected}"

    # Verify the dtype handling. The implementation logic suggests int64 inputs
    # are cast to float32 to avoid float64. We check if the result is a float type.
    assert result.dtype in [tf.float32, tf.float64], f"Expected float dtype, got {result.dtype}"

    print("Test passed: tf.keras.ops.true_divide handles int64 inputs correctly.")

if __name__ == "__main__":
    test_true_divide_int64_handling()