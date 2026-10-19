```python
import tensorflow as tf

# torch.compile is roughly equivalent to tf.function for graph compilation
@tf.function
def fn(x):
    y = x + 1
    z = x + y
    # torch._dynamo.graph_break() forces the compiler to stop tracing.
    # There is no direct equivalent in TensorFlow to force a graph break
    # inside a tf.function. The function is traced as a whole.
    # torch._dynamo.graph_break()
    return z

fn(tf.ones(3))
```