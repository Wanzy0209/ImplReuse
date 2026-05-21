import torch
import tensorflow as tf
import numpy as np

def test_tf_math_exp_with_sliced_tensor():
    """
    Test case adapted from Issue 167924 (PyTorch MPS crash on repeat_interleave with sliced tensor).
    This test preserves the reproduction logic (tensor creation and slicing) while leveraging
    the similar API (tf.math.exp) to verify behavior with sliced inputs.
    """
    # Setup from the original bug report
    # Original: counts = torch.tensor([0, 1, 0], device="mps")
    counts = tf.constant([0, 1, 0], dtype=tf.float32)
    
    # Original: data = torch.arange(2, device="mps")
    # Using tf.range as the equivalent to torch.arange
    data = tf.range(2, dtype=tf.float32)

    # The original bug involved slicing 'counts' to a non-prefix [1:3]
    # and passing it to an operation (repeat_interleave).
    # Here we apply the similar API (tf.math.exp) to this sliced tensor.
    sliced_input = counts[1:3]
    
    # Apply the similar API
    result = tf.math.exp(sliced_input)

    # Verify the result
    # sliced_input is [1, 0]
    # exp(1) = e (~2.718), exp(0) = 1
    expected = tf.constant([np.e, 1.0], dtype=tf.float32)
    
    # Assert that the operation completed without crashing and produced the correct output
    assert tf.reduce_all(tf.abs(result - expected) < 1e-6).numpy(), \
        f"Test failed: expected {expected.numpy()}, got {result.numpy()}"

    print("Test passed: tf.math.exp handled the sliced tensor correctly.")

if __name__ == "__main__":
    test_tf_math_exp_with_sliced_tensor()