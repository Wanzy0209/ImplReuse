import torch
import tensorflow as tf

def test_large_tensor_exp():
    """
    Test case adapted from PyTorch Issue 164048.
    The original issue involved an 'invalid configuration argument' when indexing
    a large tensor on CUDA. This test verifies that the similar API
    (tf.keras.ops.exp) handles large tensors on GPU without configuration errors.
    """
    # Shape from the original PyTorch reproducer
    shape = (4, 87, 1056, 736)

    # Check for GPU availability to mimic the "device='cuda'" context
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Test skipped: No GPU available.")
        return

    with tf.device('/GPU:0'):
        # Create a large tensor similar to the PyTorch reproducer
        # PyTorch: torch.randint(0, 20, ...)
        # TensorFlow: tf.random.uniform(...)
        x = tf.random.uniform(shape, minval=0, maxval=20, dtype=tf.float32)

        # Apply the similar API: tf.keras.ops.exp
        # The original bug was triggered by an operation on the large tensor.
        # We test if exp handles the large tensor configuration correctly.
        result = tf.keras.ops.exp(x)

        # Verify the operation completed successfully and properties are preserved
        assert result.shape == shape, "Output shape should match input shape for element-wise exp"
        assert result.dtype == x.dtype, "Output dtype should match input dtype"
        # Ensure the result is actually on the GPU
        assert 'GPU:0' in result.device, "Result should be on GPU"

if __name__ == "__main__":
    test_large_tensor_exp()