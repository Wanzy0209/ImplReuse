```python
import tensorflow as tf

# Conversion: @torch.compile(backend="eager") -> @tf.function
@tf.function
def fn(x, i):
    if i == 1:
        # Conversion: torch._dynamo.graph_break() is a PyTorch specific compiler directive.
        # It has no direct equivalent in TensorFlow's tf.function, so it is omitted.
        pass
    return x + 1

# Conversion: torch.randn(3) -> tf.random.normal((3,))
inp = tf.random.normal((3,))
fn(inp, 0)
fn(inp, 1)
fn(inp, 2)
```