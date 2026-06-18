import torch
import tensorflow as tf
import numpy as np

def test_tf_bitwise_not_int64_min():
    """
    Adapted test case for tf.experimental.numpy.bitwise_not based on the 
    torch.fmod crash report involving INT64_MIN.
    
    The original bug involves a crash (segfault) when processing int64 tensors 
    containing the minimum representable value. This test verifies the behavior 
    of the similar TensorFlow API with the same tensor configuration.
    """
    # Setup: Create an int64 tensor filled with the minimum representable value
    # This mirrors the 'dividend' tensor in the original PyTorch bug report.
    dividend = tf.fill((2, 3), tf.iinfo(tf.int64).min)
    dividend = tf.cast(dividend, tf.int64)

    print("Input tensor:", dividend)
    print("Input dtype:", dividend.dtype)

    # Operation: Apply the target API (tf.experimental.numpy.bitwise_not)
    # Note: bitwise_not is a unary operation (~x), unlike fmod (x % y).
    # We apply it to the edge-case tensor to check for stability/crashes.
    try:
        result = tf.experimental.numpy.bitwise_not(dividend)
        print("Result:", result)
        
        # Assertion: Verify the operation completed and returned a tensor of the expected shape
        assert result.shape == dividend.shape, "Output shape mismatch"
        assert result.dtype == dividend.dtype, "Output dtype mismatch"
        
        # Logic check: ~INT64_MIN should be INT64_MAX
        # INT64_MIN is 0x8000...0000, ~INT64_MIN is 0x7FFF...FFFF (INT64_MAX)
        expected_val = tf.iinfo(tf.int64).max
        # Check if all elements match the expected result
        assert tf.reduce_all(result == expected_val).numpy(), "Result value mismatch"
        
        print("Test passed: Operation handled INT64_MIN correctly without crashing.")
        
    except Exception as e:
        print(f"Test failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_tf_bitwise_not_int64_min()