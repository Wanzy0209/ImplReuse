import torch
import tensorflow as tf
import numpy as np

def fn(x, axes, weights):
    """
    Wrapper function for the target API: tf.compat.v1.nn.weighted_moments
    """
    # weighted_moments returns a tuple (mean, variance)
    return tf.compat.v1.nn.weighted_moments(x, axes, frequency_weights=weights)

# Generate sample inputs
# x: A 4D tensor (e.g., batch, height, width, channels)
# weights: Positive weights broadcastable to x
# axes: Axes along which to compute the moments
x = tf.random.normal((2, 4, 4, 3), dtype=tf.float32)
weights = tf.random.uniform((2, 4, 4, 3), minval=0.1, maxval=1.0, dtype=tf.float32)
axes = [1, 2]

# "Compile" the function using tf.function with JIT compilation enabled.
# This is analogous to torch.compile(..., backend="inductor", mode="max-autotune").
compiled_fn = tf.function(fn, jit_compile=True)

# 1. Run in Eager mode
mean_eager, var_eager = fn(x, axes, weights)

# 2. Run in Compiled (Graph/XLA) mode
mean_compiled, var_compiled = compiled_fn(x, axes, weights)

# 3. Assert that results are close
# This mirrors the torch.testing.assert_close check in the original bug report.
try:
    np.testing.assert_allclose(mean_eager.numpy(), mean_compiled.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(var_eager.numpy(), var_compiled.numpy(), rtol=1e-5, atol=1e-5)
    print("Test passed: Eager and compiled results are consistent.")
except AssertionError as e:
    print(f"Test failed: {e}")
    raise