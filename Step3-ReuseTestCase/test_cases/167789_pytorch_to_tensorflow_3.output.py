import torch
import sys
import tensorflow as tf

# Enable eager execution using the similar API
# Note: This must be called at program startup before any other TensorFlow operations
tf.compat.v1.enable_eager_execution()

# Set recursion limit to a high value, similar to the original bug report
sys.setrecursionlimit(10000000)

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

def outer(x):
    return fn(x, 1000)

# Execute the function
# In the original PyTorch bug, torch.compile(backend="eager") ignored the recursion limit.
# Here we verify if tf.compat.v1.enable_eager_execution respects the recursion limit.
input_tensor = tf.ones(3)
result = outer(input_tensor)

# Verify the result is correct (1 + 1000 = 1001)
expected = tf.ones(3) + 1000
assert tf.reduce_all(tf.equal(result, expected)).numpy(), "Recursion test failed"