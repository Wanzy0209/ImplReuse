```python
import tensorflow as tf

# torch._dynamo.config.nested_graph_breaks = True
# Note: TensorFlow's tf.function handles nested calls and graph breaks automatically.
# There is no direct equivalent configuration flag for nested graph breaks.

def inner(x):
    x = x + 1
    # torch._dynamo.graph_break()
    # Note: In TensorFlow, graph breaks happen implicitly when Python control flow
    # or unsupported operations are used inside a tf.function.
    return x + 2

@tf.function
# Equivalent to @torch.compile(backend="eager")
def outer(x):
    x = inner(x + 4) + 8
    return inner(x) + 16

outer(tf.ones(3))
```