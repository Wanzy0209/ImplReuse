```python
import sys
import tensorflow as tf

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

# Conversion: torch.compile(backend="eager") -> tf.function
# Note: tf.function traces the function. The recursive helper fn is kept as a Python function
# to avoid TensorFlow graph recursion limits, mimicking the eager backend behavior.
@tf.function
def outer(x):
    return fn(x, 1000)

sys.setrecursionlimit(10000000)
# Conversion: torch.ones -> tf.ones
outer(tf.ones(3))
```