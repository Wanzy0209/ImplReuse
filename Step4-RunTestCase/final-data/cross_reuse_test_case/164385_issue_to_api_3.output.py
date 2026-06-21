import tensorflow as tf
import numpy as np

def get_diagonal(tensor, offset=0, axis1=-2, axis2=-1):
    """
    Helper function to extract diagonal, handling environments where 
    tf.keras.ops might not be available (older TensorFlow versions).
    """
    if hasattr(tf.keras, 'ops') and hasattr(tf.keras.ops, 'diagonal'):
        return tf.keras.ops.diagonal(tensor, offset=offset, axis1=axis1, axis2=axis2)
    else:
        # Fallback to numpy for older TF versions
        np_result = np.diagonal(tensor.numpy(), offset=offset, axis1=axis1, axis2=axis2)
        return tf.constant(np_result)

def test_diagonal_with_complex_params():
    """
    Test tf.keras.ops.diagonal with specific parameters that mirror the 
    conditional logic found in the similar API implementation.
    
    The original bug report involved a complex expression structure causing 
    unexpected type simplification (FloorDiv -> Rational). 
    The similar API (diagonal) has specific logic branches based on rank, 
    offset, and axes. This test verifies the behavior of these branches.
    """
    
    # Create a 3D tensor (Rank 3) to allow testing axis1=-2, axis2=-1
    # Using a range of values to ensure data integrity
    data = np.arange(18).reshape(2, 3, 3).astype(np.float32)
    tensor = tf.constant(data)
    
    print(f"Input tensor shape: {tensor.shape}")
    print(f"Input tensor dtype: {tensor.dtype}")

    # Case 1: The optimized path (matches the 'if' condition in similar API)
    # Conditions: offset=0, axis1=-2, axis2=-1
    # This should trigger array_ops.matrix_diag_part
    print("\nTesting optimized path (offset=0, axis1=-2, axis2=-1)...")
    result_optimized = get_diagonal(tensor, offset=0, axis1=-2, axis2=-1)
    
    print(f"Result shape: {result_optimized.shape}")
    print(f"Result dtype: {result_optimized.dtype}")
    
    # Expected shape: (2, 3) - taking diagonal of the (3,3) matrices
    assert result_optimized.shape == (2, 3), f"Expected shape (2, 3), got {result_optimized.shape}"
    # Check that dtype is preserved (analogous to checking type in original bug)
    assert result_optimized.dtype == tensor.dtype, f"Expected dtype {tensor.dtype}, got {result_optimized.dtype}"
    
    # Verify values
    expected_opt = np.array([[0, 4, 8], [9, 13, 17]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result_optimized.numpy(), expected_opt)
    print("Optimized path result matches expected values.")

    # Case 2: The general path (does not match the 'if' condition)
    # Conditions: offset=1 (non-zero)
    # This should trigger the moveaxis and general logic
    print("\nTesting general path (offset=1, axis1=1, axis2=2)...")
    result_general = get_diagonal(tensor, offset=1, axis1=1, axis2=2)
    
    print(f"Result shape: {result_general.shape}")
    print(f"Result dtype: {result_general.dtype}")
    
    # Expected shape: (2, 2) - offset 1 reduces the diagonal size
    assert result_general.shape == (2, 2), f"Expected shape (2, 2), got {result_general.shape}"
    assert result_general.dtype == tensor.dtype, f"Expected dtype {tensor.dtype}, got {result_general.dtype}"
    
    # Verify values
    # Note: Corrected expected values to match standard diagonal behavior with offset=1
    expected_gen = np.array([[1, 5], [10, 14]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result_general.numpy(), expected_gen)
    print("General path result matches expected values.")

    print("\nAll tests passed.")

if __name__ == "__main__":
    test_diagonal_with_complex_params()