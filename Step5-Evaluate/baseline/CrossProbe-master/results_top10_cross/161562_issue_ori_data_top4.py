```python
import tensorflow as tf

# PyTorch: @torch.compile(backend="eager", fullgraph=True)
# TensorFlow: Functions run eagerly by default. @tf.function is used for graph mode,
# but since the source specifies backend="eager", we use a standard function.
def fn(x):
    y = x + 10

    # PyTorch: with torch._dynamo.set_fullgraph(False):
    # TensorFlow: This is a PyTorch Dynamo specific context manager for graph breaking.
    # It is not applicable in TensorFlow eager execution.
    class Foo:
        def __init__(self, x):
            self.x = x

    f = Foo(x)
    return f.x - y

# PyTorch: x = torch.tensor([1.0])
x = tf.constant([1.0])
y = fn(x)
print(y)
```