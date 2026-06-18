import torch
import tensorflow as tf
import numpy as np

# This test case adapts the logic from the PyTorch bug report (Issue 163929).
# The original bug involved an in-place mutation (tan_), a view (transpose),
# and a reduction (argmin) failing under torch.compile (inductor).
# Here, we translate this pattern to TensorFlow, replacing the reduction
# with the similar API: tf.compat.v1.to_complex128.
# We use tf.function to simulate the compilation/graph mode aspect.

def foo(x):
    # Equivalent to x.tan_() (in-place mutation)
    x.assign(tf.tan(x))
    # Equivalent to x = x.t() (transpose)
    x = tf.transpose(x)
    # Using the similar API instead of argmin
    return tf.compat.v1.to_complex128(x)

# Setup
np.random.seed(0)
data = np.random.randn(4, 6).astype(np.float32)

# Eager execution
var1 = tf.Variable(data)
out1 = foo(var1)

# Compiled execution (tf.function mimics torch.compile)
var2 = tf.Variable(data)
cf = tf.function(foo)
out2 = cf(var2)

# Assert that the results are consistent between eager and compiled modes
np.testing.assert_array_equal(out1.numpy(), out2.numpy())