```python
import tensorflow as tf

# Conversion: PyTorch method chaining (x.sin().cos()) maps to TensorFlow functional ops
def f(x):
    return tf.cos(tf.sin(x))

# Conversion: @torch.compile maps to @tf.function for graph execution/optimization
@tf.function
def g(x):
    # Conversion: torch.vmap vectorizes a function over a batch dimension.
    # TensorFlow ops (like sin, cos) are natively vectorized, so f(x) works directly on the batch.
    # For non-vectorizable logic, tf.vectorized_map would be the equivalent.
    return f(x)

# Conversion: torch.randn(10, device="cuda") creates random normal data.
# TensorFlow uses tf.random.normal. Device placement is automatic or via tf.device.
x = tf.random.normal((10,))
g(x)
```