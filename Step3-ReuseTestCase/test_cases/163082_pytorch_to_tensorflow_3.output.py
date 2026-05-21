import torch
import tensorflow as tf
import numpy as np

# Check for CUDA (GPU) availability
gpus = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if gpus else '/CPU:0'
print(f"Running on device: {device}")

# Define the function with compilation (equivalent to torch.compile)
@tf.function
def get_real_part_compiled(input_tensor):
    return tf.experimental.numpy.real(input_tensor)

# Define the function without compilation (eager execution)
def get_real_part_eager(input_tensor):
    return tf.experimental.numpy.real(input_tensor)

with tf.device(device):
    # Adapt the input to be complex to test the 'real' API functionality
    # Using the float values from the original bug report for the real part
    # Original: [3.799999, 0.0, 0.0]
    # Complex: [3.799999 + 1.0j, 0.0 + 1.0j, 0.0 + 0.0j]
    c = tf.constant([[3.799999 + 1.0j, 0.0 + 1.0j, 0.0 + 0.0j]], dtype=tf.complex64)

    print("Input vector (complex):", c.numpy().tolist())

    # Run compiled version
    xyz_compiled = get_real_part_compiled(c)
    print("Real part (tf.function):", xyz_compiled.numpy().tolist())

    # Run eager version
    xyz_eager = get_real_part_eager(c)
    print("Real part (eager):", xyz_eager.numpy().tolist())

    # Verify consistency (mimicking the bug check logic)
    # The original bug checked if norm > 1. Here we check if results match.
    assert np.allclose(xyz_compiled.numpy(), xyz_eager.numpy()), \
        "Discrepancy detected between compiled and eager execution for tf.experimental.numpy.real"