```python
import tensorflow as tf

class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Conversion Note: PyTorch's pytree.register_constant(Bar) is used to tell the compiler
# to treat instances of Bar as leaf nodes (constants) during graph tracing.
# TensorFlow's tf.function treats non-Tensor objects as generic Python objects automatically,
# so explicit registration is not required or supported in the public API in the same way.

@tf.function
def fn(x, obj):
    obj.attr = {3: Bar()}
    return x + 1

fn(tf.ones(3), Foo())
```