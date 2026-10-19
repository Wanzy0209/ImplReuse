```python
import tensorflow as tf

def f(x):
    # Conversion: PyTorch method chaining x.sin().cos() -> TensorFlow functional API
    # Order: x.sin() then .cos() implies cos(sin(x))
    return tf.cos(tf.sin(x))

# Conversion: @torch.compile -> @tf.function (graph compilation)
@tf.function
def g(x):
    # Conversion: torch.vmap(f)(x) -> f(x)
    # TensorFlow operations are natively vectorized, so explicit mapping is not required
    # for standard math operations. If f contained non-vectorized logic,
    # tf.vectorized_map would be the equivalent.
    return f(x)

# Conversion: torch.randn(10, device="cuda") -> tf.random.normal((10,))
# Note: TensorFlow automatically uses GPU if available and configured.
x = tf.random.normal((10,))
g(x)
```