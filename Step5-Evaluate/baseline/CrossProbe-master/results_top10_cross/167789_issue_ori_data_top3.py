```python
import sys
import tensorflow as tf

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

# torch.compile(backend="eager") executes the function in eager mode without graph compilation.
# TensorFlow runs in eager mode by default, so no decorator is applied.
def outer(x):
    return fn(x, 1000)

sys.setrecursionlimit(10000000)
# torch.ones(3) creates a 1D tensor of ones. In TensorFlow, use tf.ones.
outer(tf.ones((3,)))
```