import tensorflow as tf
import numpy as np

def test_overlap_and_add_layout_sensitivity():
    """
    Adapted from PyTorch Issue 164491.
    
    The original issue reported that `_scaled_mm` and `_int_mm` (matrix multiplication ops)
    raised errors or suffered performance degradation when the Right Hand Side (RHS) matrix
    was in row-major layout instead of column-major.
    
    This test verifies that `tf.signal.overlap_and_add` handles inputs with non-standard
    memory layouts (strides) correctly without raising errors, preserving the core logic
    of testing layout sensitivity.
    """
    
    # Setup parameters
    batch_size = 2
    frames = 10
    frame_length = 4
    frame_step = 2

    # 1. Create a standard contiguous signal
    # Shape: [batch_size, frames, frame_length]
    signal_contiguous = tf.random.uniform((batch_size, frames, frame_length), dtype=tf.float32)

    # 2. Create a signal with non-standard strides (simulating the "row-major" issue)
    # We create a larger tensor and slice it. This often results in a non-contiguous
    # memory layout (strides not equal to dimension size), analogous to the row-major
    # matrix scenario in the bug report.
    large_buffer = tf.random.uniform((batch_size, frames + 5, frame_length + 5), dtype=tf.float32)
    signal_strided = large_buffer[:, :frames, :frame_length]

    # 3. Execute the operation
    # In the PyTorch bug, passing a row-major matrix to `_scaled_mm` raised an error.
    # We verify that `tf.signal.overlap_and_add` handles the strided input gracefully.
    try:
        result_contiguous = tf.signal.overlap_and_add(signal_contiguous, frame_step)
        result_strided = tf.signal.overlap_and_add(signal_strided, frame_step)
    except Exception as e:
        print(f"API failed with strided input: {e}")
        raise AssertionError(f"tf.signal.overlap_and_add failed with non-contiguous/strided input: {e}")

    # 4. Verify correctness
    # We implement a simple numpy reference to check the strided result.
    def numpy_overlap_add(signal, step):
        output_size = (signal.shape[-2] - 1) * step + signal.shape[-1]
        output = np.zeros(signal.shape[:-2] + (output_size,), dtype=signal.dtype)
        for i in range(signal.shape[-2]):
            start = i * step
            end = start + signal.shape[-1]
            output[..., start:end] += signal[..., i, :]
        return output

    expected_strided = numpy_overlap_add(signal_strided.numpy(), frame_step)

    # Assert that the result matches the expected output.
    # This ensures the API didn't silently fail or produce garbage due to layout issues.
    np.testing.assert_allclose(result_strided.numpy(), expected_strided, rtol=1e-5)
    
    print("Test passed: tf.signal.overlap_and_add handles strided inputs correctly.")

if __name__ == "__main__":
    test_overlap_and_add_layout_sensitivity()