```python
import tensorflow as tf

# Conversion: x.sin().cos() -> tf.cos(tf.sin(x))
def f(x):
    return tf.cos(tf.sin(x))

# Conversion: @torch.compile -> @tf.function
@tf.function
def g(x):
    # Conversion: torch.vmap(f)(x) -> tf.vectorized_map(f, x)
    # Note: TensorFlow operations are natively vectorized, so calling f(x) directly
    # is often more efficient. tf.vectorized_map is used here to preserve the
    # explicit mapping logic of the original code.
    return tf.vectorized_map(f, x)

# Conversion: torch.randn(10, device="cuda") -> tf.random.normal with device context
with tf.device('/GPU:0'):
    x = tf.random.normal((10,))
    g(x)
```