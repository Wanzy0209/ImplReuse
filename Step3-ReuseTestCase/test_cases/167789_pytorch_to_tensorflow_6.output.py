import torch
import sys
import tensorflow as tf

# Adapted test case for tf.compat.v1.name_scope
# Original issue: sys.setrecursionlimit is ignored by torch.compile
# This test verifies if recursion limits are respected within tf.compat.v1.name_scope

def fn(x, n):
    if n == 0:
        return x
    # Using the similar API inside the recursive call
    with tf.compat.v1.name_scope(f"recursion_{n}"):
        return fn(x, n - 1) + 1

# Set recursion limit to a high value, mimicking the original bug report
sys.setrecursionlimit(10000000)

try:
    # In the original PyTorch code, @torch.compile wraps the execution.
    # Here, we wrap the execution in the target API context.
    with tf.compat.v1.name_scope("outer_scope"):
        # Pass a TensorFlow tensor (equivalent to torch.ones(3))
        result = fn(tf.ones(3), 1000)
    
    print("Test Passed: Recursion limit was respected.")

except RecursionError as e:
    print(f"Test Failed: RecursionError encountered despite high recursion limit: {e}")