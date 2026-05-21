import tensorflow as tf
import numpy as np

def test_saturate_cast_int8_layout_and_overflow():
    """
    Adapts the logic from the PyTorch bug report regarding _int_mm and row-major matrices.
    
    The original bug highlights issues with integer matrix multiplication (_int_mm) 
    when the Right Hand Side (rhs) matrix is in a row-major layout (often resulting 
    from transposing a column-major weight matrix). The operation was either slow 
    or raised errors depending on the layout.

    For the similar API tf.dtypes.saturate_cast, we verify:
    1. Correct handling of integer overflow (saturation), which is critical for 
       low-precision operations like those mentioned in the bug (int8).
    2. Robustness against non-contiguous memory layouts (transposed tensors), 
       mirroring the "row-major" sensitivity in the original bug.
    """
    
    # --- Test Case 1: Saturation Logic (Overflow/Underflow) ---
    # The PyTorch bug involves _int_mm, implying int8 operations.
    # We verify that saturate_cast correctly clamps values to the int8 range [-128, 127].
    print("Testing saturation logic for int8...")
    
    # Values designed to overflow int8
    raw_values = np.array([1000, -1000, 0, 127, -128, 128, -129], dtype=np.int32)
    input_tensor = tf.constant(raw_values)
    
    # Perform saturating cast to int8
    result_tensor = tf.dtypes.saturate_cast(input_tensor, tf.int8)
    
    # Expected clamped values
    expected_values = np.array([127, -128, 0, 127, -128, 127, -128], dtype=np.int8)
    
    # Assert correctness
    assert tf.reduce_all(tf.equal(result_tensor, expected_values)).numpy(), \
        f"Saturation failed. Got {result_tensor.numpy()}, expected {expected_values}"
    print("Saturation logic test passed.")

    # --- Test Case 2: Layout Sensitivity (Transposed/Row-Major) ---
    # The original bug reported that _scaled_mm raised an error and _int_mm was slow 
    # with row-major rhs matrices. We verify saturate_cast handles non-contiguous 
    # (transposed) tensors without error.
    print("Testing behavior on transposed (non-contiguous) tensors...")

    # Create a matrix with values that will saturate
    # Shape (2, 3)
    matrix_data = np.array([
        [1000, 2000, 3000], 
        [-1000, -2000, -3000]
    ], dtype=np.int32)
    
    input_matrix = tf.constant(matrix_data)
    
    # Transpose the matrix. In many frameworks, this changes the memory stride/layout.
    # The original bug specifically mentioned issues with transposed weights (row-major).
    transposed_matrix = tf.transpose(input_matrix) # Shape becomes (3, 2)
    
    # Cast the transposed matrix
    result_transposed = tf.dtypes.saturate_cast(transposed_matrix, tf.int8)
    
    # Manually calculate expected result for the transposed view
    # Original:
    # [ 1000,  2000,  3000]
    # [-1000, -2000, -3000]
    # Transposed:
    # [ 1000, -1000]
    # [ 2000, -2000]
    # [ 3000, -3000]
    # Saturated (int8):
    # [ 127, -128]
    # [ 127, -128]
    # [ 127, -128]
    expected_transposed = np.array([
        [127, -128],
        [127, -128],
        [127, -128]
    ], dtype=np.int8)
    
    assert tf.reduce_all(tf.equal(result_transposed, expected_transposed)).numpy(), \
        f"Transposed tensor cast failed. Got {result_transposed.numpy()}, expected {expected_transposed}"
    print("Transposed tensor test passed.")

if __name__ == "__main__":
    test_saturate_cast_int8_layout_and_overflow()
    print("All tests passed successfully.")