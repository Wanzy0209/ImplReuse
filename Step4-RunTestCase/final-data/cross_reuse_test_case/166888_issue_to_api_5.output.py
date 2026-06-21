import torch
import tensorflow as tf
import tf.experimental.numpy as tnp

# Test case adapted from PyTorch Issue 166888
# Original Bug: torch.compile fails when .item() is called on a tensor arg inside a function (e.g., for clamp).
# Similar API: tf.experimental.numpy.ndim (extracts scalar rank from tensor).
# Adaptation: Use tf.experimental.numpy.ndim inside a tf.function to extract a scalar and use it in an operation.

def test_ndim_in_compiled_context():
    # Enable numpy behavior for tf.experimental.numpy
    tnp.experimental_enable_numpy_behavior()

    @tf.function
    def f(x):
        # Extract scalar using the similar API (ndim)
        # This mirrors max_val.item() in the original bug, where a scalar is derived from a tensor argument
        rank = tnp.ndim(x)

        # Use the scalar in an operation (clip_by_value mirrors torch.clamp)
        # We cast rank to float32 to match the tensor dtype for the operation
        y = tf.clip_by_value(x, 0.0, tf.cast(rank, tf.float32))
        return y

    # Create input tensor
    x = tf.random.normal((10, 20, 30))

    # Execute the compiled function
    # This checks if the compilation handles the scalar extraction correctly
    result = f(x)

    # Assertions to verify correctness
    assert result.shape == x.shape
    # Since rank is 3, values should be clamped between 0 and 3
    assert tf.reduce_all(result >= 0.0)
    assert tf.reduce_all(result <= 3.0)

if __name__ == "__main__":
    test_ndim_in_compiled_context()
    print("Test passed.")