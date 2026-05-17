import tensorflow as tf
import numpy as np

# Reconstructing the similar API based on the provided snippet
# to ensure the test is runnable and reflects the code pattern.
def with_space_to_batch(
    input,  # pylint: disable=redefined-builtin
    dilation_rate,
    padding,
    op,
    filter_shape=None,
    spatial_dims=None,
    data_format=None):
    """
    Performs `op` on the space-to-batch representation of `input`.
    Adapted from the provided snippet for testing purposes.
    """
    # Simplified logic for the test case:
    # 1. Pad input
    # 2. Space to Batch
    # 3. Apply Op
    # 4. Batch to Space
    
    # Assuming 2D spatial dims for this test
    # padding format: [[0,0], [top, bottom], [left, right], [0,0]]
    padded_input = tf.pad(input, padding)
    
    # dilation_rate acts as block_shape
    s2b = tf.space_to_batch_nd(padded_input, dilation_rate, [[0, 0], [0, 0]])
    
    # Apply op
    # The op signature in the snippet is op(input, num_spatial_dims, padding)
    # We pass the transformed input
    res = op(s2b, 2, "VALID")
    
    # Batch to space
    # crops = [[0,0], [0,0]] (simplified)
    output = tf.batch_to_space_nd(res, dilation_rate, [[0, 0], [0, 0]])
    
    return output

def test_complex_tensor_consistency():
    """
    Test case adapted from the PyTorch index_add bug report.
    Verifies that the similar API (with_space_to_batch) preserves
    complex tensor components (Real and Imaginary) correctly.
    """
    # Setup: Create a complex tensor with non-zero imaginary parts
    # Shape: [Batch, Height, Width, Channels]
    shape = [1, 4, 4, 1]
    
    # Real part: ones
    real_part = np.ones(shape, dtype=np.float32)
    # Imaginary part: random non-zero values
    imag_part = np.random.randn(*shape).astype(np.float32)
    
    input_tensor = tf.complex(real_part, imag_part)
    
    # Define a simple operation (Identity) to test the pipeline's data integrity
    def identity_op(inp, num_spatial_dims, padding):
        return inp
    
    # Parameters
    dilation_rate = [2, 2]
    # Padding to ensure divisibility (4 is divisible by 2, so 0 padding works)
    padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
    
    # Execute
    result = with_space_to_batch(input_tensor, dilation_rate, padding, identity_op)
    
    # Verification
    # The PyTorch bug showed that the imaginary sum became 0.0 on the failing backend.
    # We check that the imaginary part is preserved (sum is not 0).
    imag_sum = tf.reduce_sum(tf.abs(tf.math.imag(result))).numpy()
    real_sum = tf.reduce_sum(tf.abs(tf.math.real(result))).numpy()
    
    print(f"Real sum: {real_sum}")
    print(f"Imaginary sum: {imag_sum}")
    
    # Assert that the imaginary part is not zeroed out
    assert imag_sum > 1e-6, f"Imaginary part lost (sum: {imag_sum})"
    
    # Also verify the real part is handled correctly
    assert real_sum > 1e-6, f"Real part lost (sum: {real_sum})"
    
    print("Test passed: Complex tensor components preserved.")

if __name__ == "__main__":
    test_complex_tensor_consistency()