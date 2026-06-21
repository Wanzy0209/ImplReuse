import tensorflow as tf
from tensorflow.experimental import numpy as tnp

# Check for CUDA (GPU) availability
gpus = tf.config.list_physical_devices('GPU')
has_cuda = len(gpus) > 0

def test_diagonal(device_name: str = '/CPU:0'):
    """
    Test if tf.experimental.numpy.diagonal works correctly when tensors 
    are placed on a specific device (CPU or CUDA).
    """
    # TensorFlow uses context managers for device placement rather than a global set_default_device
    with tf.device(device_name):
        # Create a random matrix (similar to creating tensors in the original bug report)
        # Using tf.experimental.numpy to mimic the numpy-like environment
        x = tnp.random.randn(100, 100)

        # Call the target API: tf.experimental.numpy.diagonal
        # This corresponds to the 'random_split' call in the original bug
        try:
            result = tnp.diagonal(x)
            
            # Verify the result shape is correct
            assert result.shape == (100,), f"Expected shape (100,), got {result.shape}"
            
            print(f"Device {device_name} worked.")
        except Exception as e:
            print(f"Device {device_name} failed with error: {e}")
            raise

# Test on CPU (should always work)
test_diagonal(device_name='/CPU:0')

# Test on CUDA (GPU) if available
if has_cuda:
    test_diagonal(device_name='/GPU:0')
else:
    print("CUDA device not available. Skipping GPU test.")