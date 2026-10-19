```python
import tensorflow as tf

# TensorFlow's tf.function treats non-Tensor arguments (like class instances) as constants implicitly.
# There is no direct public equivalent to torch.utils._pytree.register_constant.

class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# pytree.register_constant(Bar) is not required in TensorFlow

@tf.function
def fn(x, obj):
    obj.attr = {3: Bar()}
    return x + 1

fn(tf.ones(3), Foo())
```