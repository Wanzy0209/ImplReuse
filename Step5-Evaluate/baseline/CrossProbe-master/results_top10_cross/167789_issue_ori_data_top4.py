```python
import sys
import tensorflow as tf

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

# torch.compile(backend="eager") indicates execution without graph compilation.
# In TensorFlow 2.x, eager execution is the default, so we define outer as a standard function.
def outer(x):
    return fn(x, 1000)

sys.setrecursionlimit(10000000)
# torch.ones(3) creates a tensor of ones with shape (3,)
outer(tf.ones(3))
```