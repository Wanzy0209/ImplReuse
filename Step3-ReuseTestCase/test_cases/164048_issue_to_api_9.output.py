import torch
import tensorflow as tf

def test_large_tensor_exp():
    """
    Test case for tf.math.exp derived from Issue 164048.
    
    The original issue involved a crash ("invalid configuration argument") when 
    performing indexing on a large tensor on CUDA. This test adapts the 
    "large tensor" context to the similar API (tf.math.exp) to ensure it 
    handles high-dimensional inputs correctly.
    """
    # Dimensions from the original bug report
    # Original: mask = torch.randint(0, 20, (4, 87, 1056, 736), device="cuda")
    shape = (4, 87, 1056, 736)

    # Create a large tensor. 
    # Note: tf.math.exp requires floating point inputs, so we use float32.
    # We map the original integer range [0, 20] to floats.
    large_tensor = tf.random.uniform(shape, minval=0.0, maxval=20.0, dtype=tf.float32)

    # Apply the similar API: tf.math.exp
    # This replaces the indexing operation from the original bug report
    result = tf.math.exp(large_tensor)

    # Assertions
    # 1. Verify the operation completed and shape is preserved (element-wise operation)
    assert result.shape == shape, f"Shape mismatch: expected {shape}, got {result.shape}"
    
    # 2. Verify mathematical property (exp(x) > 0 for real x)
    assert tf.reduce_all(result > 0).numpy(), "Exponential result should be positive for real inputs"

    print("Test passed: tf.math.exp handled large tensor successfully.")

if __name__ == "__main__":
    test_large_tensor_exp()