import tensorflow as tf
import pytest

def test_keras_ops_trace_on_device():
    """
    Test case for tf.keras.ops.trace adapted from the PyTorch MPS issue.
    
    The original issue (Issue ID: 163962) involves performing a linear algebra 
    operation (PARAFAC decomposition) on a tensor placed on the MPS device.
    This test mirrors that logic by creating a tensor, placing it on an available 
    device (GPU or CPU), and performing a similar linear algebra operation (trace)
    using the tf.keras.ops API.
    """
    # Create a tensor. 
    # Note: The original issue used a 3D tensor (12, 3, 12). 
    # tf.keras.ops.trace operates on matrices (or batches of matrices), 
    # so we adapt the shape to be a square matrix (12, 12) to fit the API semantics.
    x = tf.ones((12, 12))

    # Mimic the device placement logic from the original bug report (.to("mps")).
    # We attempt to use the GPU if available, otherwise fallback to CPU.
    # Added compatibility handling for different TensorFlow versions.
    try:
        # Try TensorFlow 2.x method
        has_gpu = bool(tf.config.list_physical_devices('GPU'))
    except AttributeError:
        # Fallback for TensorFlow 1.x or environments where list_physical_devices is missing
        try:
            has_gpu = tf.test.is_gpu_available()
        except (AttributeError, NotImplementedError):
            has_gpu = False

    device_name = '/GPU:0' if has_gpu else '/CPU:0'

    with tf.device(device_name):
        # Perform the operation using the similar API
        # This corresponds to the 'parafac' call in the original issue.
        result = tf.keras.ops.trace(x)

    # Assertion to verify the operation executed correctly.
    # The trace of a 12x12 matrix of ones is 12.
    expected = 12.0
    assert result == expected, f"Expected trace to be {expected}, but got {result} on {device_name}"

if __name__ == "__main__":
    test_keras_ops_trace_on_device()
    print("Test passed.")