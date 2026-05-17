import tensorflow as tf
import numpy as np

def test_to_double_consistency():
    """
    Adapted from Issue 161014: Inconsistent constant_pad_nd behavior with negative padding.
    
    The original bug involved `constant_pad_nd` throwing a RuntimeError on valid inputs
    (specifically when mixing negative padding with positive padding), resulting in 
    incorrect shape validation errors.
    
    This test verifies that `tf.compat.v1.to_double` handles various valid input 
    configurations (different dtypes, shapes, and arguments) consistently without 
    raising unexpected errors, and correctly preserves the tensor shape.
    """
    
    # Case 1: Basic usage (Analogous to the first working example in the bug report)
    # Input: float32 tensor
    # Expected: float64 tensor with same shape
    input_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    output_tensor = tf.compat.v1.to_double(input_tensor)
    
    assert output_tensor.dtype == tf.float64, "Output dtype should be float64"
    assert output_tensor.shape == input_tensor.shape, "Shape should be preserved"
    np.testing.assert_array_equal(output_tensor.numpy(), input_tensor.numpy())

    # Case 2: Different input type (Analogous to the second working example)
    # Input: int32 tensor
    # Expected: float64 tensor with same shape
    input_int = tf.constant([1, 2, 3], dtype=tf.int32)
    output_int = tf.compat.v1.to_double(input_int)
    
    assert output_int.dtype == tf.float64, "Output dtype should be float64"
    assert output_int.shape == input_int.shape, "Shape should be preserved"
    np.testing.assert_array_equal(output_int.numpy(), [1.0, 2.0, 3.0])

    # Case 3: Usage with extra arguments (Analogous to the failing example in the bug report)
    # The bug report showed that adding extra padding arguments caused a crash.
    # Here we add the 'name' argument to ensure the API handles it robustly.
    input_named = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    
    try:
        output_named = tf.compat.v1.to_double(input_named, name="my_cast_operation")
        assert output_named.dtype == tf.float64, "Output dtype should be float64"
        assert output_named.shape == input_named.shape, "Shape should be preserved"
        np.testing.assert_array_equal(output_named.numpy(), input_named.numpy())
    except Exception as e:
        # This mirrors the "RuntimeError" in the original bug. 
        # If this occurs, the API is inconsistent.
        raise AssertionError(f"tf.compat.v1.to_double failed unexpectedly with valid arguments: {e}")

if __name__ == "__main__":
    test_to_double_consistency()
    print("Test passed successfully.")