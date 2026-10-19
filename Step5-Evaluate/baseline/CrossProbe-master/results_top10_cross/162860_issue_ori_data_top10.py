```python
import tensorflow as tf

def inner(x):
    return x + 1

# Conversion: torch.compile maps to tf.function
# Note: backend="eager" in PyTorch implies eager execution, 
# but tf.function is the standard equivalent for decorating a callable for execution.
@tf.function
def fn(x):
    x = inner(x)
    return inner(x)

# Conversion: torch.ones maps to tf.ones
fn(tf.ones(3))
```