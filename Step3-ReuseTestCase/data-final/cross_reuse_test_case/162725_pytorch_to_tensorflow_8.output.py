import torch
import tensorflow as tf
import numpy as np

def fn(x1, x2):
    # Target API: tf.experimental.numpy.logaddexp
    return tf.experimental.numpy.logaddexp(x1, x2)

# Generate sample inputs to simulate the op_db sampling in the original bug report.
# We include various shapes and edge cases (NaNs, Infs) relevant to logaddexp.
sample_inputs = [
    (tf.random.normal((2, 3, 4), dtype=tf.float32), tf.random.normal((2, 3, 4), dtype=tf.float32)),
    (tf.constant([1.0, 100.0, 1000.0]), tf.constant([1.0, 100.0, 1000.0])),
    (tf.constant([float('inf'), -float('inf')]), tf.constant([1.0, 1.0])),
    (tf.constant([float('nan'), 0.0]), tf.constant([0.0, float('nan')])),
]

# Compile the function using tf.function with XLA (jit_compile=True)
# to mimic the compilation backend optimization (like inductor) in the original bug.
compiled_fn = tf.function(fn, jit_compile=True)

for x1, x2 in sample_inputs:
    # Eager execution
    res1 = fn(x1, x2)
    
    # Compiled execution
    res2 = compiled_fn(x1, x2)
    
    # Verify consistency between eager and compiled results
    # This mirrors the torch.testing.assert_close in the original issue
    try:
        tf.debugging.assert_all_close(res1, res2, rtol=1e-5, atol=1e-5)
        print(f"Test passed for inputs shape {x1.shape}")
    except tf.errors.InvalidArgumentError as e:
        print(f"Test failed for inputs shape {x1.shape}")
        print(f"Eager result: {res1}")
        print(f"Compiled result: {res2}")
        raise e