```python
import tensorflow as tf
import contextlib

flag = True

# Conversion: @torch.compile(backend="eager") -> @tf.function
@tf.function
def fn(x):
    x = x + 1
    # Conversion: torch._dynamo.graph_break() -> tf.py_function forces a graph break
    tf.py_function(lambda: None, [], [])
    x = x + 2
    if flag:
        # Conversion: with torch.no_grad() -> contextlib.nullcontext
        # (TF uses explicit GradientTape, so no_grad is implicit in forward pass)
        with contextlib.nullcontext():
            # Conversion: torch._dynamo.graph_break() -> tf.py_function
            tf.py_function(lambda: None, [], [])
    else:
        with contextlib.nullcontext():
            tf.py_function(lambda: None, [], [])
    return x + 4


# Conversion: torch.ones(3) -> tf.ones(3)
fn(tf.ones(3))
flag = False
fn(tf.ones(3))
```