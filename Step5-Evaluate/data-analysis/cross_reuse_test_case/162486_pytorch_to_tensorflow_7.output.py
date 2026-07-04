import tensorflow as tf
from tensorflow.keras.ops import slice

def test_slice_on_device(device_name: str = '/GPU:0'):
    """
    Adapts the PyTorch random_split test to use tf.keras.ops.slice.
    Instead of splitting a dataset object, we slice the underlying tensors
    to verify behavior on different devices.
    """
    # TensorFlow uses context managers for device placement rather than 
    # a global set_default_device, but the effect is similar for the scope.
    with tf.device(device_name):
        # Create data similar to the PyTorch example
        x = tf.random.normal((100, 3))
        y = tf.random.normal((100, 2))

        # Mimic the split logic: taking the first 70% (70 rows)
        # PyTorch: random_split(dataset, [0.7, 0.2, 0.1])
        # TensorFlow: slice(x, start, limit, strides)
        start_dims = [0, 0]
        limit_dims = [70, 3] # 70% of 100 is 70
        strides = [1, 1]

        # Perform the slice operation
        x_sliced = slice(x, start_dims, limit_dims, strides)

        # Basic assertion to verify the operation worked and shape is correct
        assert x_sliced.shape == (70, 3), f"Expected shape (70, 3), got {x_sliced.shape}"

        print(f"Device {device_name} worked.")

# Test on CPU
test_slice_on_device('/CPU:0')

# Test on GPU (if available)
if tf.config.list_physical_devices('GPU'):
    test_slice_on_device('/GPU:0')
else:
    print("GPU not available, skipping GPU test.")