```python
import sys
import tensorflow as tf

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

# Conversion: torch.compile(backend="eager") -> tf.function
# tf.function compiles the function into a graph. AutoGraph will convert the recursion.
@tf.function
def outer(x):
    return fn(x, 1000)

sys.setrecursionlimit(10000000)
# Conversion: torch.ones(3) -> tf.ones(3)
outer(tf.ones(3))
```