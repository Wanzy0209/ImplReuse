```python
import sys
import tensorflow as tf

# Conversion: torch.compile is translated to tf.function for graph optimization.
# Note: The provided context for torch.compile (random_ops_util) is not applicable 
# to the logic of this specific snippet, so standard tf.function is used.
def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

@tf.function
def outer(x):
    return fn(x, 1000)

sys.setrecursionlimit(10000000)
outer(tf.ones(3))
```