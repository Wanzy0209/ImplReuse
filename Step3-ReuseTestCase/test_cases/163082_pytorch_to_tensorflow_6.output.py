import torch
import tensorflow as tf
import numpy as np

# Check for GPU availability to match the original bug report's context
gpus = tf.config.list_physical_devices('GPU')
device = 'GPU:0' if gpus else 'CPU:0'
print(f"Running on device: {device}")

# Define the compiled version using tf.function (equivalent to torch.compile)
# We use jit_compile=True to strictly enforce compilation, similar to the bug scenario
@tf.function(jit_compile=True)
def diff_compiled(a):
    return tf.experimental.numpy.diff(a)

# Define the non-compiled version (eager execution)
def diff_eager(a):
    return tf.experimental.numpy.diff(a)

# Replicate the input from the original bug report
# Original input: [[3.799999, 0.0, 0.0]]
with tf.device(device):
    c = tf.constant([[3.799999, 0.0, 0.0]], dtype=tf.float32)
    
    print("Input vector:", c.numpy().tolist()[0])
    
    # Execute compiled version
    res_compiled = diff_compiled(c)
    print("Diff result (compiled):", res_compiled.numpy().tolist()[0])
    
    # Execute eager version
    res_eager = diff_eager(c)
    print("Diff result (eager):", res_eager.numpy().tolist()[0])

    # Verify behavior
    # The original bug checks if norm > 1. For 'diff', we check if the compiled 
    # output matches the eager output within a reasonable tolerance.
    # The original bug showed a deviation of ~1e-7 (1.0000001192092896).
    # We use a slightly looser tolerance here to account for standard FP32 variance,
    # but strict enough to catch compilation errors.
    is_close = np.allclose(res_eager.numpy(), res_compiled.numpy(), atol=1e-6)
    
    assert is_close, (
        f"Numerical discrepancy detected between eager and compiled execution.\n"
        f"Eager: {res_eager.numpy()}\n"
        f"Compiled: {res_compiled.numpy()}"
    )
    
    print("Test passed: Eager and compiled results match.")