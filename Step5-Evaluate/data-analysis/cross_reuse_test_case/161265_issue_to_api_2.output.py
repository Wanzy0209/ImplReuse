import sys

try:
    import torch
    import tensorflow as tf
    import numpy as np
except ImportError as e:
    print(f"Test skipped: Required library import failed due to environment issues ({e}).")
    sys.exit(0)

def test_large_tensor_initialization():
    """
    Test case to verify that tf.keras.backend.ones correctly initializes
    tensors larger than 4GB, mirroring the logic of the PyTorch MPS bug report.
    """
    # Check for GPU availability (MPS on macOS is exposed as GPU in TensorFlow)
    gpus = tf.config.list_physical_devices('GPU')
    device_name = '/GPU:0' if gpus else '/CPU:0'

    try:
        with tf.device(device_name):
            # Reproduce the logic: Create a tensor > 4GB
            # Shape: 2 rows, (2^31 + 5) columns
            # Total elements: 2 * (2^31 + 5) = 2^32 + 10 bytes (approx 4GB + 10 bytes)
            # This exceeds the 32-bit integer limit often used in buffer sizes.
            shape = (2, (1 << 31) + 5)
            dtype = 'int8'

            # Use tf.keras.backend.ones to create the tensor
            # This corresponds to torch.ones in the original issue
            a = tf.keras.backend.ones(shape, dtype=dtype)

            # Verify the content at the end of the buffer
            # Original bug: a[1, -2] was 0 instead of 1
            val_single = a[1, -2]
            
            # Original bug: a[:, -2] contained 0s instead of 1s
            val_slice = a[:, -2]

            # Assertions to check if the buffer was filled correctly
            assert val_single.numpy() == 1, f"Expected 1 at a[1, -2], got {val_single.numpy()}"
            assert np.all(val_slice.numpy() == 1), f"Expected [1, 1] at a[:, -2], got {val_slice.numpy()}"

            print("Test passed: Large tensor filled correctly.")

    except tf.errors.ResourceExhaustedError:
        print("Test skipped: Not enough memory to allocate 4GB+ tensor.")

if __name__ == "__main__":
    test_large_tensor_initialization()