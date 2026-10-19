```python
import tensorflow as tf

# torch._dynamo.config.nested_graph_breaks = True
# Note: TensorFlow's tf.function does not have a direct configuration equivalent for nested graph breaks.

def inner(x):
    x = x + 1
    # torch._dynamo.graph_break()
    # Note: TensorFlow does not support explicit graph breaks within a tf.function.
    # To execute Python code eagerly inside a graph, tf.py_function can be used,
    # but it requires explicit input/output shapes and dtypes.
    return x + 2

# @torch.compile(backend="eager")
# Note: @tf.function is used to compile a TensorFlow function into a graph.
@tf.function
def outer(x):
    x = inner(x + 4) + 8
    return inner(x) + 16

# outer(torch.ones(3))
outer(tf.ones(3))
```