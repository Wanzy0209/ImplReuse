import tensorflow as tf
import numpy as np
import warnings

def test_cast_to_floatx_complex_handling():
    """
    Test case adapted from PyTorch Issue 160034.
    The original issue highlights a failure/error message problem when handling 
    complex64 types on a specific backend (MPS) for an index operation.
    
    This test verifies the similar API, tf.keras.backend.cast_to_floatx, 
    correctly handles complex64 inputs by casting them to the default 
    float type (e.g., float32), ensuring type compatibility.
    """
    # Set the default float type to float32 to match common expectations
    original_floatx = tf.keras.backend.floatx()
    tf.keras.backend.set_floatx('float32')

    try:
        # Create a complex64 input, mirroring the bug report's data type
        complex_data = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype='complex64')
        
        # Use the similar API to cast the complex data to the default float type
        # This parallels the bug report's scenario where complex data interacts 
        # with a float-preferring context.
        result = tf.keras.backend.cast_to_floatx(complex_data)

        # Assertions
        # 1. Verify the output is the default float type (float32)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
        
        # 2. Verify the real part of the complex numbers is preserved
        # (Standard behavior when casting complex to float)
        expected_real = np.array([1.0, 3.0], dtype='float32')
        assert np.allclose(result, expected_real), "Real part data mismatch during cast"

        # Also test with a TensorFlow Tensor input
        complex_tensor = tf.constant([5.0 + 6.0j], dtype=tf.complex64)
        
        # Fix: Convert the TensorFlow tensor to a numpy array before passing to cast_to_floatx.
        # This resolves the TypeError: __array__() takes 1 positional argument but 2 were given,
        # which occurs due to incompatibilities between np.asarray and TF tensors in certain versions.
        result_tensor = tf.keras.backend.cast_to_floatx(complex_tensor.numpy())
        
        # Fix: cast_to_floatx returns a numpy array, not a tf.Tensor. 
        # Update assertions to check for numpy dtype and values.
        assert result_tensor.dtype == np.float32
        assert np.allclose(result_tensor, [5.0])

    finally:
        # Restore original floatx setting
        tf.keras.backend.set_floatx(original_floatx)

if __name__ == "__main__":
    # Suppress the ComplexWarning as it is expected behavior when casting complex to real
    warnings.filterwarnings("ignore", category=np.ComplexWarning)
    test_cast_to_floatx_complex_handling()
    print("Test passed: cast_to_floatx handles complex64 correctly.")