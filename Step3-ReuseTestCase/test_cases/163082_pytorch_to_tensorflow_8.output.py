import torch
import tensorflow as tf
import numpy as np

# Check for GPU availability to mimic the original 'cuda' setup
gpus = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if gpus else '/CPU:0'

print(f"Running on device: {device}")

# Define the compiled function (TensorFlow equivalent to @torch.compile)
@tf.function
def fix_compiled(x):
    return tf.experimental.numpy.fix(x)

# Define the uncompiled function
def fix_uncompiled(x):
    return tf.experimental.numpy.fix(x)

# Original input vector from the bug report
input_data = [[3.799999, 0.0, 0.0]]

with tf.device(device):
    c = tf.constant(input_data, dtype=tf.float32)
    print("Input vector:", c.numpy().tolist()[0])

    # Run compiled version
    xyz_compiled = fix_compiled(c)
    print("Output (compiled):", xyz_compiled.numpy().tolist()[0])

    # Run uncompiled version
    xyz_uncompiled = fix_uncompiled(c)
    print("Output (uncompiled):", xyz_uncompiled.numpy().tolist()[0])

    # Verify behavior
    # The original bug was about precision drift (norm > 1).
    # For tf.experimental.numpy.fix, we verify that the compiled version
    # matches the uncompiled version exactly.
    assert np.allclose(xyz_compiled.numpy(), xyz_uncompiled.numpy()), \
        "Mismatch between compiled and uncompiled outputs"
    
    print("Test passed: Compiled and uncompiled outputs match.")