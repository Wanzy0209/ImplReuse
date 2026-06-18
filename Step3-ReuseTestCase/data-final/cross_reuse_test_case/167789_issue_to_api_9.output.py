import sys
import tensorflow as tf

# The similar API extracted is tf.autograph.trace (logging during tracing).
# We leverage this API to observe the execution flow during the tracing phase,
# mirroring the debugging context of the original bug report.

def fn(x, n):
    # Use the similar API to trace the recursion depth
    tf.autograph.trace(f"Recursion depth: {n}")
    
    if n == 0:
        return x
    return fn(x, n - 1) + 1

@tf.function
def outer(x):
    return fn(x, 1000)

# Reproduce the original bug logic: set a high recursion limit
# and attempt to execute a deep recursive call within a compiled context.
sys.setrecursionlimit(10000000)

try:
    result = outer(tf.ones(3))
    # If the library handles the recursion limit correctly (or converts to loop),
    # this assertion passes. The original bug was a RecursionError despite the limit.
    assert result.shape == (3,)
    print("Test passed: Recursion limit respected or recursion handled.")
except RecursionError:
    print("Test failed: RecursionError hit despite sys.setrecursionlimit.")