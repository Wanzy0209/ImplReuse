import torch
import numpy as np

# Handle environment incompatibility issues (e.g., missing GLIBCXX version)
try:
    import tensorflow as tf
except ImportError as e:
    print("Test skipped: Failed to import TensorFlow due to environment issues.")
    print(f"Error details: {e}")
    print("This is likely due to a missing GLIBCXX version (e.g., GLIBCXX_3.4.29) required by protobuf/tensorflow.")
    import sys
    sys.exit(0)

def test_floor_divide_type_promotion():
    """
    Test case for tf.keras.ops.floor_divide based on PyTorch Issue 164465.
    
    The original issue involves a crash with int64 dtype during compilation (torch.compile).
    The similar API (tf.keras.ops.floor_divide) contains logic for dtype promotion and 
    specific handling for bool -> int8 casting.
    
    This test verifies that floor_divide correctly handles int64 inputs (mimicking the bug's 
    context) and bool inputs (mimicking the API's implementation details) within a 
    compiled context (tf.function).
    """

    # 1. Test with int64 (mimicking the bug report's dtype and sequence generation)
    # PyTorch: torch.ops.prims.iota.default(..., dtype=torch.int64)
    @tf.function
    def compile_int64_divide():
        # Create a sequence of int64 values (equivalent to iota/arange)
        x = tf.range(10, dtype=tf.int64)
        y = tf.constant(3, dtype=tf.int64)
        
        # Perform floor_divide (the similar API)
        return tf.keras.ops.floor_divide(x, y)

    result_int64 = compile_int64_divide()
    expected_int64 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3], dtype=np.int64)
    
    assert result_int64.dtype == tf.int64, "Expected int64 dtype"
    assert np.array_equal(result_int64.numpy(), expected_int64), "Int64 floor_divide calculation failed"

    # 2. Test with bool (mimicking the API implementation's specific logic)
    # The API code snippet shows: if x1.dtype == dtypes.bool: cast(x1, dtypes.int8)
    @tf.function
    def compile_bool_divide():
        x = tf.constant([True, False, True, False])
        # Avoid division by zero
        y = tf.constant([True, True, True, True]) 
        
        return tf.keras.ops.floor_divide(x, y)

    result_bool = compile_bool_divide()
    
    # Based on the snippet, bools are cast to int8 before division
    # True(1) // True(1) = 1, False(0) // True(1) = 0
    expected_bool = np.array([1, 0, 1, 0], dtype=np.int8)
    
    assert result_bool.dtype == tf.int8, "Expected int8 dtype for bool inputs"
    assert np.array_equal(result_bool.numpy(), expected_bool), "Bool floor_divide calculation failed"

    print("Test passed: floor_divide handles int64 and bool type promotion correctly.")

if __name__ == "__main__":
    test_floor_divide_type_promotion()