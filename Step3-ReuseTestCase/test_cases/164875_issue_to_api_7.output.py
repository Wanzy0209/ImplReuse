import tensorflow as tf
import numpy as np

def test_assert_none_equal_zero_dim_divergence():
    """
    Test case for tf.compat.v1.assert_none_equal based on Issue 164875.
    
    The original issue reported an eager/compile divergence in PyTorch when 
    performing operations on tensors with shape (20, 0). This test verifies 
    that the similar TensorFlow API handles these edge-case shapes consistently 
    in both eager and graph (compiled) modes.
    """
    
    # Recreate the specific shape scenario from the PyTorch bug report
    # Shape (20, 0) represents a non-zero batch size with a zero feature dimension
    shape = (20, 0)
    
    # Create tensors with the problematic shape
    # We use int64 to match the dtype in the original bug report
    x = tf.zeros(shape, dtype=tf.int64)
    y = tf.ones(shape, dtype=tf.int64)

    print("Testing tf.compat.v1.assert_none_equal with shape (20, 0)")

    # 1. Test in Eager Mode
    # In the original bug, eager mode succeeded.
    try:
        tf.compat.v1.assert_none_equal(x, y)
        print(' eager success')
    except Exception as e:
        print(f' eager failed: {e}')

    # 2. Test in Compiled Mode (tf.function)
    # In the original bug, compiled mode failed with a size mismatch error.
    # We use tf.function to simulate the compilation/tracing step.
    @tf.function
    def compiled_assert(x_in, y_in):
        return tf.compat.v1.assert_none_equal(x_in, y_in)

    try:
        compiled_assert(x, y)
        print(' compile success')
    except Exception as e:
        print(f' compile failed: {e}')

if __name__ == "__main__":
    test_assert_none_equal_zero_dim_divergence()