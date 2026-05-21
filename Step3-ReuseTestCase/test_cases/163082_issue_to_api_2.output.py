import torch
import tensorflow as tf
import numpy as np

# Set seed for reproducibility, mirroring the original bug report
tf.random.set_seed(1337)

# Define the compiled version using tf.function (equivalent to torch.compile)
# We use jit_compile=True to match the aggressive compilation context of the bug
@tf.function(jit_compile=True)
def hinge_loss_jit(y_true, y_pred):
    return tf.keras.losses.hinge(y_true, y_pred)

# Define the non-compiled version
def hinge_loss_no_jit(y_true, y_pred):
    return tf.keras.losses.hinge(y_true, y_pred)

# Determine device (GPU if available, to match the 'cuda' context in the bug report)
device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print(f"Running on device: {device_name}")

with tf.device(device_name):
    # Create inputs. 
    # The original bug used specific float32 values [3.799999, 0.0, 0.0].
    # For hinge loss, we test a boundary condition where y_true * y_pred is close to 1.
    # If y_true=1 and y_pred=1, loss is max(0, 1-1) = 0.
    # We use float32 explicitly as the bug was specific to fp32 precision.
    y_true = tf.constant([[1.0]], dtype=tf.float32)
    y_pred = tf.constant([[1.0]], dtype=tf.float32)
    
    print("Input y_true:", y_true.numpy().flatten())
    print("Input y_pred:", y_pred.numpy().flatten())

    # Run compiled version
    loss_jit = hinge_loss_jit(y_true, y_pred)
    print("Hinge loss (jit_compile):", loss_jit.numpy().flatten())

    # Run non-compiled version
    loss_no_jit = hinge_loss_no_jit(y_true, y_pred)
    print("Hinge loss (no jit):", loss_no_jit.numpy().flatten())

    # Assertions to check for precision issues.
    # The original bug checked for norm > 1. For hinge loss, the equivalent violation is loss < 0.
    # We also check if the JIT version differs from the non-JIT version.
    assert loss_jit.numpy()[0] >= 0.0, f"JIT compiled loss is negative: {loss_jit.numpy()[0]} (precision error)"
    assert np.allclose(loss_jit.numpy(), loss_no_jit.numpy(), atol=1e-6), \
        f"JIT and non-JIT results differ: JIT={loss_jit.numpy()}, NoJIT={loss_no_jit.numpy()}"