```python
import tensorflow as tf

# Conversion: @torch.compile(backend="eager") -> @tf.function
# Note: PyTorch's graph_break forces a fallback to eager execution.
# TensorFlow's tf.function handles control flow natively within the graph.
@tf.function
def fn(x, i):
    # Conversion: torch._dynamo.graph_break()
    # No direct equivalent in TensorFlow to force a graph break mid-execution.
    # The conditional logic is preserved.
    if i == 1:
        pass
    return x + 1

# Conversion: torch.randn(3) -> tf.random.normal((3,))
inp = tf.random.normal((3,))
fn(inp, 0)
fn(inp, 1)
fn(inp, 2)
```