import torch
import tensorflow as tf

def test_he_normal_redundant_dtype_conversion():
    """
    Test case derived from PyTorch Issue 161611.
    
    This test verifies that using tf.keras.initializers.HeNormal with a specific dtype
    renders a subsequent cast to the same dtype redundant, mirroring the issue
    found in the scaled_dot_product_attention docstring where:
    1. A tensor is created with a specific dtype.
    2. A redundant operation attempts to convert it to that same dtype.
    """
    # Mimic the 'query' tensor and its dtype from the original bug
    query_dtype = tf.float32
    shape = (10, 10)

    # Original: attn_bias = torch.zeros(..., dtype=query.dtype, device=query.device)
    # Similar API: Initialize a tensor using HeNormal with the specific dtype
    initializer = tf.keras.initializers.HeNormal()
    attn_bias = initializer(shape=shape, dtype=query_dtype)

    # Verify the dtype is correct immediately after initialization
    assert attn_bias.dtype == query_dtype, \
        f"Expected dtype {query_dtype}, but got {attn_bias.dtype}"

    # Original: attn_bias.to(query.dtype) # Redundant!
    # Similar logic: Attempting to cast the tensor to the dtype it already has.
    # In TensorFlow, tf.cast returns a new tensor. If the dtype matches, 
    # this operation is redundant and computationally wasteful.
    redundant_cast = tf.cast(attn_bias, query_dtype)

    # Assertions to confirm the redundancy
    assert redundant_cast.dtype == query_dtype, \
        "Redundant cast changed the dtype unexpectedly"
    
    # Verify that the redundant cast did not alter the values (identity operation)
    # Note: We use tf.equal to check value identity since HeNormal generates random values.
    assert tf.reduce_all(tf.equal(attn_bias, redundant_cast)).numpy(), \
        "Redundant cast changed the tensor values"

if __name__ == "__main__":
    test_he_normal_redundant_dtype_conversion()
    print("Test passed: Redundant dtype conversion confirmed as no-op.")