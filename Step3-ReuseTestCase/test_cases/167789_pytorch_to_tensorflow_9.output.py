import torch
import sys
import tensorflow as tf

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

# Adaptation: Using tf.name_scope as the wrapper similar to torch.compile
# Note: tf.name_scope is primarily a context manager for naming, but can be used as a decorator.
@tf.name_scope("outer_scope")
def outer(x):
    return fn(x, 1000)

sys.setrecursionlimit(10000000)

# Execute the test
# In the original PyTorch bug, this raises RecursionError.
# Here, we expect it to succeed, verifying that tf.name_scope respects the recursion limit.
try:
    result = outer(tf.ones(3))
    # Verify the result is correct (1 + 1000 = 1001)
    assert tf.reduce_all(result == 1001).numpy()
    print("Test passed: Recursion limit respected.")
except RecursionError:
    print("Test failed: RecursionError raised despite high limit.")
    raise