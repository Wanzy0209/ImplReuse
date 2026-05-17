import torch
import tensorflow as tf

def test_multiply_no_nan_redundant_dtype_conversion():
    """
    Adapted from PyTorch Issue #161611.
    
    This test verifies that tf.math.multiply_no_nan preserves the dtype of its inputs,
    making a subsequent cast to the same dtype redundant (similar to the redundant
    .to() call in the PyTorch bug report).
    """
    # Define a target dtype (mimicking query.dtype in the original bug)
    target_dtype = tf.float32

    # Create inputs with the target dtype
    # Mimics: attn_bias = torch.zeros(..., dtype=query.dtype, ...)
    x = tf.constant([1.0, 2.0, 3.0], dtype=target_dtype)
    y = tf.constant([4.0, 0.0, 6.0], dtype=target_dtype)

    # Perform the operation
    # Mimics the logic flow where attn_bias is used/modified
    result = tf.math.multiply_no_nan(x, y)

    # The "Bug" pattern: Redundant dtype conversion
    # Original: attn_bias.to(query.dtype) # Redundant! Already query.dtype
    # Adapted: tf.cast(result, target_dtype)
    # This is redundant because result.dtype is already target_dtype
    redundant_result = tf.cast(result, target_dtype)

    # Verification 1: Result already has the correct dtype
    # This proves that the cast is redundant
    assert result.dtype == target_dtype, \
        f"Expected dtype {target_dtype}, but got {result.dtype}"

    # Verification 2: The redundant cast produces the same values
    assert tf.reduce_all(tf.equal(result, redundant_result)), \
        "Redundant cast changed the tensor values"

    # Verification 3: Verify the specific behavior of multiply_no_nan
    # (2.0 * 0.0 should be 0.0, not NaN)
    expected_values = [4.0, 0.0, 18.0]
    assert tf.reduce_all(tf.equal(result, expected_values)), \
        "Operation did not produce expected values"

    print("Test passed: Redundant dtype conversion verified as unnecessary.")

if __name__ == "__main__":
    test_multiply_no_nan_redundant_dtype_conversion()