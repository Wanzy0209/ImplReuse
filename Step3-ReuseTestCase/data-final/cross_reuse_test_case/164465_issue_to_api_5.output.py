import torch
import tensorflow as tf
import numpy as np

# The original issue involved a crash with torch.compile, int64, and sequence operations.
# This test case targets the similar API (tf.experimental.numpy.lcm) using
# tf.function (analogous to torch.compile) and int64 sequences to verify
# correct behavior and stability.

@tf.function
def test_lcm_with_int64_sequences(a, b):
    # Using the similar API
    return tf.experimental.numpy.lcm(a, b)

# Create int64 sequences similar to the 'iota' in the original bug report
# Original: torch.ops.prims.iota.default(36, ..., dtype=torch.int64)
x = tf.range(0, 36, dtype=tf.int64)
y = tf.range(1, 37, dtype=tf.int64)

# Execute the compiled function
result = test_lcm_with_int64_sequences(x, y)

# Verify the result against numpy's implementation
expected = np.lcm(x.numpy(), y.numpy())
assert np.array_equal(result.numpy(), expected), "LCM calculation mismatch"

print("Test passed.")