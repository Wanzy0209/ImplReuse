```python
import tensorflow as tf

def inner(x):
    return x + 1

# torch.compile(backend="eager") indicates that the function should run in eager mode.
# TensorFlow 2.x runs in eager mode by default, so no @tf.function decorator is applied here.
def fn(x):
    x = inner(x)
    return inner(x)

fn(tf.ones(3))
```