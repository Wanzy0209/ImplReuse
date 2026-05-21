import tensorflow as tf
import sys

# Preserving the original bug reproduction logic: 
# A recursive function structure similar to the PyTorch issue.
def fn(x, n):
    if n == 0:
        return x
    
    # Leveraging the similar API (tf.errors.OperatorNotAllowedInGraphError):
    # We introduce an operation (iterating over a tensor) that is valid in 
    # Python/Eager mode but raises OperatorNotAllowedInGraphError in Graph mode.
    # This mirrors the "Graph mode restriction" theme of the original issue.
    for _ in x:
        pass
        
    return fn(x, n - 1)

@tf.function
def outer(x):
    return fn(x, 3)

# Test execution
# We expect the call to fail with OperatorNotAllowedInGraphError because
# the graph mode does not support iterating over a tensor directly.
try:
    outer(tf.ones(3))
    raise AssertionError("Expected tf.errors.OperatorNotAllowedInGraphError")
except tf.errors.OperatorNotAllowedInGraphError as e:
    print("Test passed: Caught expected error:", e)