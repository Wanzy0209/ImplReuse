import tensorflow as tf

# Enable eager execution to ensure the compat v1 API runs immediately
tf.compat.v1.enable_eager_execution()

def test_tf_dropout_with_2d_input():
    """
    Adapted test case for tf.compat.v1.nn.dropout based on the PyTorch EmbeddingBag bug report.
    
    Original Bug: EmbeddingBag with include_last_offset=True generates incorrect offsets 
    ([0, 4] instead of [0, 4, 8]) for 2D input.
    
    Adaptation: Since tf.compat.v1.nn.dropout does not have an 'include_last_offset' 
    parameter or handle indices/offsets like EmbeddingBag, we verify that the API 
    correctly handles the 2D input structure described in the bug report without error.
    """
    
    # Original input from the bug report (indices)
    # EmbeddingBag expects Long (int), but Dropout expects Float.
    # We adapt the values to float to test the Dropout API.
    input_data = tf.constant([[1.0, 2.0, 4.0, 5.0], [4.0, 3.0, 2.0, 9.0]], dtype=tf.float32)
    
    # Parameters for the similar API
    # The provided snippet for tf.compat.v1.nn.dropout requires a seed.
    rate = 0.5
    seed = 42
    
    # Call the API
    # Note: There is no 'include_last_offset' flag in dropout to test.
    try:
        output = tf.compat.v1.nn.dropout(input_data, rate=rate, seed=seed)
    except Exception as e:
        print(f"API call failed with error: {e}")
        raise

    # Verify behavior
    # 1. Check output shape matches input shape (standard for dropout)
    assert output.shape == input_data.shape, \
        f"Shape mismatch: expected {input_data.shape}, got {output.shape}"
    
    # 2. Check output dtype
    assert output.dtype == input_data.dtype, \
        f"Dtype mismatch: expected {input_data.dtype}, got {output.dtype}"

    # 3. Verify values are either 0 or scaled by 1/(1-rate)
    # This confirms the core logic of dropout is applied to the 2D input.
    scale = 1.0 / (1.0 - rate)
    
    # Create a mask of where values are not zero
    non_zero_mask = tf.not_equal(output, 0.0)
    
    # Extract non-zero values
    non_zero_values = tf.boolean_mask(output, non_zero_mask)
    
    # Check if non-zero values are approximately equal to input * scale
    # (We check a subset or use reduce_all if the tensor is small enough)
    expected_non_zero = tf.boolean_mask(input_data * scale, non_zero_mask)
    
    # Assert that the non-zero values are scaled correctly
    # Using a small tolerance for floating point comparison
    tf.debugging.assert_near(non_zero_values, expected_non_zero, rtol=1e-5, atol=1e-5)

    print("Test passed: tf.compat.v1.nn.dropout handled the 2D input correctly.")

if __name__ == "__main__":
    test_tf_dropout_with_2d_input()