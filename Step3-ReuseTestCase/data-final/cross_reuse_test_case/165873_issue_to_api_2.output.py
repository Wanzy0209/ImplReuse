import tensorflow as tf
import numpy as np

# The original bug report highlights a scenario where a shape mismatch
# (loading a 1D tensor into a scalar Parameter) was silently ignored.
# This test case checks if tf.experimental.numpy.trace handles a similar
# shape mismatch (passing a 1D tensor to a function expecting 2D) correctly.

def test_trace_shape_mismatch():
    # Create a 1D tensor, analogous to the 'large_tensor' in the bug report
    large_tensor_1d = tf.random.normal((32000,))

    # tf.experimental.numpy.trace is expected to mimic numpy.trace,
    # which requires a 2-D array (or N-D where the last two axes are used).
    # Passing a 1-D array is a shape mismatch.
    # We expect a ValueError to be raised, ensuring strict shape checking
    # unlike the silent failure observed in the PyTorch bug.

    try:
        result = tf.experimental.numpy.trace(large_tensor_1d)
        # If no error is raised, this might indicate a similar "silent failure"
        # or permissive behavior.
        print(f"Result: {result}")
        assert False, "Expected ValueError for 1-D input to trace"
    except ValueError as e:
        # Correct behavior: raising an error for shape mismatch
        print(f"Correctly raised ValueError: {e}")

if __name__ == "__main__":
    test_trace_shape_mismatch()