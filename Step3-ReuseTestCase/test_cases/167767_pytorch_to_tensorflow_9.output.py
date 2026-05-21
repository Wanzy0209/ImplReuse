import torch
import tensorflow as tf
import numpy as np

def verify_bounds(tensor, min_val, max_val):
    """Helper to verify that tensor values respect the provided bounds."""
    vals = tensor.numpy()
    if min_val is not None:
        assert np.all(vals >= min_val), f"Lower bound violation: found values < {min_val}"
    if max_val is not None:
        # Allow for slight floating point inaccuracies if max is finite
        if np.isfinite(max_val):
            assert np.all(vals <= max_val), f"Upper bound violation: found values > {max_val}"

# 1. Corresponds to a.clamp(min=0.0)
# In TF, we generate values with minval=0.0
print("Test 1: minval=0.0")
a = tf.raw_ops.RandomUniform(shape=[1], minval=0.0, maxval=1.0, dtype=tf.float32, seed=42)
print(a)
verify_bounds(a, 0.0, 1.0)

# 2. Corresponds to b.clamp(min=1e-7)
print("\nTest 2: minval=1e-7")
b = tf.raw_ops.RandomUniform(shape=[1], minval=1e-7, maxval=1.0, dtype=tf.float32, seed=42)
print(b)
verify_bounds(b, 1e-7, 1.0)

# 3. Corresponds to b.clamp(min=1e-7, max=None)
# TF RandomUniform requires maxval. We use a large number to simulate 'None' (unbounded above).
print("\nTest 3: minval=1e-7, maxval=large (simulating None)")
b = tf.raw_ops.RandomUniform(shape=[1], minval=1e-7, maxval=1e9, dtype=tf.float32, seed=42)
print(b)
verify_bounds(b, 1e-7, 1e9)

# 4. Corresponds to b.clamp(min=1e-7, max=torch.inf)
print("\nTest 4: minval=1e-7, maxval=inf")
b = tf.raw_ops.RandomUniform(shape=[1], minval=1e-7, maxval=float('inf'), dtype=tf.float32, seed=42)
print(b)
verify_bounds(b, 1e-7, float('inf'))

# 5. Corresponds to b.clamp_min(1e-7)
print("\nTest 5: minval=1e-7 (clamp_min equivalent)")
b = tf.raw_ops.RandomUniform(shape=[1], minval=1e-7, maxval=1.0, dtype=tf.float32, seed=42)
print(b)
verify_bounds(b, 1e-7, 1.0)