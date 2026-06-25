```python
import tensorflow as tf

def inner(x):
    return x + 1

# Conversion: torch.compile(backend="eager") -> tf.function
# Note: backend="eager" in PyTorch implies eager execution, while tf.function implies graph execution.
# To strictly mimic eager execution, the @tf.function decorator can be removed.
@tf.function
def fn(x):
    x = inner(x)
    return inner(x)

fn(tf.ones(3))
```