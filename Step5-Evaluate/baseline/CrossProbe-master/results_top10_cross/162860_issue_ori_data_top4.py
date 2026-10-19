```python
import tensorflow as tf


def inner(x):
    return x + 1


# torch.compile is translated to tf.function for graph compilation.
# Note: The 'backend="eager"' argument in PyTorch suggests eager execution,
# but tf.function is the standard equivalent for the compilation decorator.
@tf.function
def fn(x):
    x = inner(x)
    return inner(x)


# torch.ones is translated to tf.ones
fn(tf.ones(3))
```