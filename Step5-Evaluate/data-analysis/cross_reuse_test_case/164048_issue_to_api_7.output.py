import tensorflow as tf

def test_large_tensor_exp():
    """
    Test case for tf.keras.backend.exp based on the environment of Issue 164048.
    The original issue involved an 'invalid configuration argument' when indexing
    a large tensor on CUDA. This test verifies that the similar API (exp) handles
    large tensors on the target device (GPU if available) without crashing.
    """
    # Dimensions from the original bug report
    shape = (4, 87, 1056, 736)

    # Check for GPU availability to mimic the original "device='cuda'" condition
    # Handle compatibility for different TensorFlow versions (TF 2.1+ vs TF 1.x/2.0)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        device_name = '/GPU:0' if gpus else '/CPU:0'
    except AttributeError:
        # Fallback for older TensorFlow versions where list_physical_devices is not available
        device_name = '/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'

    with tf.device(device_name):
        # Create a large tensor. 
        # Note: tf.keras.backend.exp operates on floats, unlike the integer mask in the bug.
        # We use float32 to ensure the API is applicable.
        large_tensor = tf.random.uniform(shape, minval=0.0, maxval=20.0, dtype=tf.float32)

        # Apply the similar API: tf.keras.backend.exp
        # This tests if the operation handles the large tensor configuration correctly.
        result = tf.keras.backend.exp(large_tensor)

        # Assertions to verify correctness and successful execution
        assert result.shape == large_tensor.shape, "Output shape mismatch"
        assert result.dtype == tf.float32, "Output dtype mismatch"
        
        print(f"Test passed on {device_name} with shape {shape}.")

if __name__ == "__main__":
    test_large_tensor_exp()