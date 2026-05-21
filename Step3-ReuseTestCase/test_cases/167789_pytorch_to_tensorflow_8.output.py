import torch
import sys
import tensorflow as tf

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

# Set recursion limit high to attempt to avoid RecursionError
sys.setrecursionlimit(10000000)

# Use the similar API: tf.keras.name_scope
# Unlike torch.compile, name_scope is a context manager, so we wrap the execution.
with tf.keras.name_scope("recursion_test"):
    # Create input tensor
    x = tf.ones(3)
    # Execute the recursive function
    # In the PyTorch bug, this raises RecursionError despite the high limit.
    # Here we verify that the recursion limit is respected.
    result = fn(x, 1000)

# Verify the result to ensure the recursion completed successfully
# Expected result: 1.0 (initial) + 1000 (increments) = 1001.0
expected = tf.constant(1001.0, dtype=tf.float32)
assert tf.reduce_all(tf.equal(result, expected)).numpy(), "Test failed: Recursion did not complete correctly"

print("Test passed: Recursion limit respected within tf.keras.name_scope.")