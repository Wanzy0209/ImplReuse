import tensorflow as tf
import numpy as np

def test_large_tensor_non_neg_constraint():
    """
    Test case adapted from PyTorch Issue 164048.
    The original issue involved an 'invalid configuration argument' when indexing
    a large tensor with a boolean mask on CUDA.
    
    This test verifies the similar API (tf.keras.constraints.NonNeg) which 
    internally uses boolean logic (math_ops.greater_equal) to process tensors.
    We ensure that applying this constraint to a large tensor of similar dimensions
    executes without error and produces the correct result.
    """
    # Define dimensions matching the original bug report
    # Shape: (4, 87, 1056, 736)
    shape = (4, 87, 1056, 736)
    
    # Create a large tensor with random values, including negatives to test the constraint
    # Using float32 as it is the default floatx for Keras
    large_tensor = tf.random.uniform(shape, minval=-10.0, maxval=10.0, dtype=tf.float32)
    
    # Instantiate the similar API: NonNeg
    # This API performs: w * cast(greater_equal(w, 0), floatx())
    # This mirrors the boolean operation pattern found in the original issue.
    constraint = tf.keras.constraints.NonNeg()
    
    # Apply the constraint to the large tensor
    # This tests the robustness of the boolean operation on large tensors
    result = constraint(large_tensor)
    
    # Assertions to verify correctness
    # 1. The shape should remain unchanged
    assert result.shape == shape, f"Shape mismatch: expected {shape}, got {result.shape}"
    
    # 2. All values should be non-negative (the core logic of NonNeg)
    # This verifies that the internal boolean mask (greater_equal) was applied correctly
    is_non_neg = tf.reduce_all(result >= 0.0)
    assert is_non_neg.numpy(), "Constraint failed: tensor contains negative values"
    
    # 3. Verify that positive values are preserved (or scaled if applicable, though NonNeg is 0 or 1 scaling)
    # NonNeg sets negatives to 0, keeps positives as is.
    # We check a random sample to ensure the operation wasn't a no-op or total wipe
    mask = large_tensor >= 0.0
    expected_result = large_tensor * tf.cast(mask, tf.float32)
    # Use reduce_all to check exact equality of the operation
    assert tf.reduce_all(result == expected_result).numpy(), "Result values do not match expected NonNeg logic"

if __name__ == "__main__":
    test_large_tensor_non_neg_constraint()
    print("Test passed.")