import torch
import tensorflow as tf
import tf.experimental.numpy as tnp

def test_negative_dtype_redundancy():
    """
    Adapted test case based on Issue 161611.
    
    Original Issue: Redundant dtype conversion in PyTorch docstring example.
    Original API: torch.zeros
    Similar API: tf.experimental.numpy.negative
    
    This test verifies that tf.experimental.numpy.negative preserves the input dtype,
    making any subsequent redundant conversion (like tf.cast to the same dtype) unnecessary,
    similar to the logic described in the original bug report.
    """
    
    # Setup: Define a target dtype (mimicking query.dtype in the original bug)
    target_dtype = tf.float32
    
    # Create an input tensor with the target dtype
    # In the original bug, torch.zeros was used to create attn_bias with query.dtype
    input_tensor = tnp.asarray([1.0, 2.0, 3.0], dtype=target_dtype)
    
    # Apply the similar API: tf.experimental.numpy.negative
    # This corresponds to the creation and modification of attn_bias in the original snippet
    result_tensor = tnp.negative(input_tensor)
    
    # Verification 1: Check if the API preserves the dtype correctly
    # The original bug highlighted that attn_bias was already query.dtype
    assert result_tensor.dtype == target_dtype, (
        f"Expected output dtype to be {target_dtype}, but got {result_tensor.dtype}. "
        "If dtypes match, a redundant cast is unnecessary."
    )
    
    # Verification 2: Simulate the redundant conversion from the bug report
    # Original code: attn_bias.to(query.dtype) # Redundant!
    # TensorFlow equivalent: tf.cast(result_tensor, target_dtype)
    redundant_cast_result = tf.cast(result_tensor, target_dtype)
    
    # Verification 3: Ensure the redundant operation is indeed ineffective (no change)
    # The original bug noted the result of .to() was not assigned, but even if assigned,
    # it is redundant if dtypes match.
    assert tf.reduce_all(tf.equal(result_tensor, redundant_cast_result)).numpy(), \
        "Redundant cast should not modify tensor values."
    assert redundant_cast_result.dtype == target_dtype, \
        "Redundant cast should not modify tensor dtype."
        
    print("Test passed: tf.experimental.numpy.negative preserves dtype, making redundant casts unnecessary.")

if __name__ == "__main__":
    test_negative_dtype_redundancy()