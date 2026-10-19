```python
import tensorflow as tf


def inner(x):
    return x + 1


# Conversion: torch.compile(backend="eager") -> tf.function
# Note: tf.function compiles the function into a static graph, analogous to torch.compile.
@tf.function
def fn(x):
    x = inner(x)
    return inner(x)


# Conversion: torch.ones(3) -> tf.ones(3)
fn(tf.ones(3))
```