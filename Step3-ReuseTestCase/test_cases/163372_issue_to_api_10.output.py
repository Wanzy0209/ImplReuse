import torch
import tensorflow as tf
import numpy as np

def test_logspace_memory_and_broadcasting():
    """
    Test case for tf.keras.ops.logspace inspired by Issue 163372.
    
    The original issue involved torch.expand being misinterpreted as torch.repeat,
    leading to excessive memory allocation during compilation. This test verifies
    that tf.keras.ops.logspace handles large tensor generation and broadcasting
    (analogous to expansion) efficiently under compilation (tf.function).
    """
    
    # Parameters mimicking the scale of the original bug report
    NUM_SAMPLES = 5000
    START = 1.0
    STOP = 10.0
    BASE_SCALAR = 10.0
    
    # 1. Test basic functionality with large size
    # This checks if the API can handle the size that caused issues in the bug report
    result = tf.keras.ops.logspace(START, STOP, num=NUM_SAMPLES, base=BASE_SCALAR)
    assert result.shape == (NUM_SAMPLES,), f"Expected shape ({NUM_SAMPLES},), got {result.shape}"

    # 2. Test under compilation (tf.function)
    # The bug specifically occurred with torch.compile. We use tf.function to test
    # graph compilation behavior.
    @tf.function(jit_compile=True)
    def compiled_logspace(start, stop, num, base):
        return tf.keras.ops.logspace(start, stop, num=num, base=base)

    result_compiled = compiled_logspace(START, STOP, NUM_SAMPLES, BASE_SCALAR)
    assert result_compiled.shape == (NUM_SAMPLES,)
    
    # Verify numerical correctness
    expected = np.logspace(START, STOP, num=NUM_SAMPLES, base=BASE_SCALAR)
    np.testing.assert_allclose(result_compiled.numpy(), expected, rtol=1e-5)

    # 3. Test Broadcasting behavior (Internal "Expand" check)
    # The implementation of logspace uses math_ops.pow(base, result).
    # If 'base' is a tensor, broadcasting occurs. We verify this doesn't
    # cause unexpected memory expansion (like the repeat vs expand bug).
    # Here we pass a base tensor of shape (2,) to broadcast against (5000,).
    base_tensor = tf.constant([10.0, 2.0]) 
    
    @tf.function(jit_compile=True)
    def compiled_logspace_broadcast(start, stop, num, base):
        return tf.keras.ops.logspace(start, stop, num=num, base=base)

    result_broadcast = compiled_logspace_broadcast(START, STOP, NUM_SAMPLES, base_tensor)
    
    # Expected shape is (2, 5000) due to broadcasting
    expected_shape = (2, NUM_SAMPLES)
    assert result_broadcast.shape == expected_shape, \
        f"Expected broadcasted shape {expected_shape}, got {result_broadcast.shape}"

    print("Test passed: tf.keras.ops.logspace handles large sizes and broadcasting correctly under compilation.")

if __name__ == "__main__":
    test_logspace_memory_and_broadcasting()