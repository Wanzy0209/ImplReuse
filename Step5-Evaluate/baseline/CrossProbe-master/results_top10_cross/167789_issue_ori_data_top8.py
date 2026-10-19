```python
import sys
import tensorflow as tf

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

# Conversion: torch.compile maps to tf.function for graph execution/compilation
@tf.function
def outer(x):
    return fn(x, 1000)

sys.setrecursionlimit(10000000)

# Conversion: torch.ones maps to tf.ones
outer(tf.ones(3))
```