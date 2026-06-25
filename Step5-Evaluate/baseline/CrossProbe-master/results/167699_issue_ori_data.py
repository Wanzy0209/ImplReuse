```python
import tensorflow as tf

# Conversion: x.sin().cos() -> tf.sin(tf.cos(x))
def f(x):
    return tf.sin(tf.cos(x))

# Conversion: @torch.compile -> @tf.function
# Conversion: torch.vmap(f)(x) -> f(x)
# Note: TensorFlow operations are vectorized by default, so explicit mapping
# (like tf.vectorized_map) is not needed for standard math operations.
@tf.function
def g(x):
    return f(x)

# Conversion: torch.randn(10, device="cuda") -> tf.random.normal((10,)) on GPU
with tf.device('/GPU:0'):
    x = tf.random.normal((10,))
    g(x)
```