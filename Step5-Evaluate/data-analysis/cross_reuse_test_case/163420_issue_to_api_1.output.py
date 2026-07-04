import torch
import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    # Handle environment issues such as GLIBCXX version mismatch
    print(f"Skipping test: Failed to import TensorFlow due to environment incompatibility.")
    print(f"Error details: {e}")
    import sys
    sys.exit(0)

def test_geomspace_0d_tensors():
    """
    Test case for tf.experimental.numpy.geomspace based on the PyTorch bug report.
    
    The original bug (Issue 163420) involves a failure when handling 0-d tensors
    (scalars) and extracting their values to perform operations (fill_diagonal_)
    within a compiled context.
    
    This test verifies that the similar API, tf.experimental.numpy.geomspace,
    correctly handles 0-d tensor inputs (start, stop) and scalar parameters (num),
    ensuring robustness in type promotion and scalar handling similar to the
    context of the original bug.
    """
    # Mimic the 0-d tensor inputs from the PyTorch bug
    # PyTorch: arg1 = torch.empty([], dtype=torch.float32)
    start = tf.constant(1.0, dtype=tf.float32, shape=[]) # Explicit 0-d tensor
    stop = tf.constant(10.0, dtype=tf.float32, shape=[])   # Explicit 0-d tensor
    num = 5

    # Call the similar API
    # The implementation logic (dtype promotion, asarray, scalar math)
    # mirrors the complexity of handling scalar inputs in the original bug.
    result = tf.experimental.numpy.geomspace(start, stop, num=num)

    # Assertions
    assert result.shape == (num,), f"Expected shape ({num},), got {result.shape}"
    
    # Verify values against standard numpy implementation
    expected = np.geomspace(1.0, 10.0, num=num)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    print("Test Passed: geomspace handles 0-d tensor inputs correctly.")

if __name__ == "__main__":
    test_geomspace_0d_tensors()