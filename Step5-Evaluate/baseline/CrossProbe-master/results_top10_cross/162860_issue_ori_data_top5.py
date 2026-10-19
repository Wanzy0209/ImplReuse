```python
import tensorflow as tf

def inner(x):
    return x + 1

# Conversion: torch.compile(backend="eager") runs the function in eager mode.
# TensorFlow 2.x runs eagerly by default, so no decorator is needed to match this behavior.
def fn(x):
    x = inner(x)
    return inner(x)

# Conversion: torch.ones(3) creates a tensor filled with 1s
fn(tf.ones(3))
```