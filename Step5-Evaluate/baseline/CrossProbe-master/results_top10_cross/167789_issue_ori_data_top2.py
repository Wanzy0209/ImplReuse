```python
import sys
import tensorflow as tf

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

# Conversion: torch.compile is translated to tf.function to enable graph execution
@tf.function
def outer(x):
    return fn(x, 1000)

sys.setrecursionlimit(10000000)
# Conversion: torch.ones is translated to tf.ones
outer(tf.ones(3))
```