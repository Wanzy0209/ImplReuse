```python
import tensorflow as tf
from contextlib import nullcontext

# torch.compile is translated to tf.function to enable graph execution.
# fullgraph=True implies strict graph capture, which is the default for tf.function.
@tf.function
def fn(x):
    y = x + 10

    # torch._dynamo.set_fullgraph(False) is a PyTorch specific context manager.
    # TensorFlow does not have a direct equivalent for toggling graph capture modes
    # dynamically inside a function trace. We use nullcontext to preserve structure.
    with nullcontext():
        class Foo:
            def __init__(self, x):
                self.x = x

    f = Foo(x)
    return f.x - y

# torch.tensor is translated to tf.constant.
x = tf.constant([1.0])
y = fn(x)
print(y)
```