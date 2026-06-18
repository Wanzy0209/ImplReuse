import torch
import tensorflow as tf
import numpy as np

# Check for GPU availability to mimic the 'cuda' condition
gpus = tf.config.list_physical_devices('GPU')

def test_tf_slice(device: str):
    """
    Adapts the PyTorch random_split logic to TensorFlow's tf.slice.
    The original bug involved splitting a dataset when the default device was CUDA.
    Here, we verify that tf.slice correctly partitions tensors on the specified device.
    """
    # In TensorFlow, we use context managers to set the device,
    # analogous to torch.set_default_device.
    with tf.device(device):
        # Create data analogous to the PyTorch dataset tensors
        # PyTorch: x = torch.randn(100, 3), y = torch.randn(100, 2)
        x = tf.random.normal((100, 3))
        y = tf.random.normal((100, 2))

        # PyTorch random_split splits the dataset into [0.7, 0.2, 0.1] of the total length.
        # Total length is 100. Splits are 70, 20, 10.
        # We use tf.slice to achieve this partitioning on the tensors directly.
        
        # Split 1: First 70 elements
        # tf.slice(input, begin, size)
        x_split1 = tf.slice(x, [0, 0], [70, 3])
        y_split1 = tf.slice(y, [0, 0], [70, 2])

        # Split 2: Next 20 elements (start at index 70)
        x_split2 = tf.slice(x, [70, 0], [20, 3])
        y_split2 = tf.slice(y, [70, 0], [20, 2])

        # Split 3: Last 10 elements (start at index 90)
        x_split3 = tf.slice(x, [90, 0], [10, 3])
        y_split3 = tf.slice(y, [90, 0], [10, 2])

        # Verify the splits have the correct shapes
        assert x_split1.shape == (70, 3), f"Expected shape (70, 3), got {x_split1.shape}"
        assert x_split2.shape == (20, 3), f"Expected shape (20, 3), got {x_split2.shape}"
        assert x_split3.shape == (10, 3), f"Expected shape (10, 3), got {x_split3.shape}"
        
        # Verify the splits have the correct types
        assert x_split1.dtype == tf.float32
        assert y_split1.dtype == tf.float32

        print(f"Device {device} worked.")

# Run the test on CPU (Baseline)
test_tf_slice(device='/CPU:0')

# Run the test on GPU if available (Analogous to the 'cuda' bug trigger)
if gpus:
    test_tf_slice(device='/GPU:0')
else:
    print("No GPU found, skipping GPU test.")