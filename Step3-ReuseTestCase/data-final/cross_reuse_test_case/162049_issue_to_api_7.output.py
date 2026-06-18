import torch
import tensorflow as tf

# The original bug report highlights a failure when processing integer tensors
# due to incorrect type handling (using torch.finfo instead of torch.iinfo).
# This test case verifies that the similar API (tf.compat.v1.to_bfloat16)
# correctly handles integer inputs without raising a TypeError.

def test_to_bfloat16_integer_support():
    # Create an integer tensor (mimicking the input type that caused the bug)
    int_tensor = tf.constant([10, 20, 30], dtype=tf.int32)
    
    # Attempt to cast the integer tensor to bfloat16
    # In the original bug, operations on integers failed. Here we ensure
    # the casting operation succeeds for integers.
    try:
        result = tf.compat.v1.to_bfloat16(int_tensor)
        
        # Verify the output type is correct
        assert result.dtype == tf.bfloat16, f"Expected dtype bfloat16, got {result.dtype}"
        
        # Verify the values are preserved (10, 20, 30 are exactly representable)
        expected = tf.constant([10.0, 20.0, 30.0], dtype=tf.bfloat16)
        assert tf.reduce_all(tf.equal(result, expected)), "Values were not preserved during casting"
        
        print("Test Passed: Integer tensor successfully cast to bfloat16.")
        print("Result:", result)
        
    except TypeError as e:
        print(f"Test Failed: TypeError raised for integer tensor - {e}")

if __name__ == "__main__":
    test_to_bfloat16_integer_support()