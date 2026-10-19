```python
import tensorflow as tf

# Conversion: torch.set_default_dtype -> tf.keras.backend.set_floatx
tf.keras.backend.set_floatx('float64')

# Conversion: device="cuda" -> tf.device context
with tf.device('/GPU:0'):
    # Conversion: torch.randn -> tf.random.normal
    # Explicit dtype is passed to ensure float64 is used
    x = tf.random.normal((1000, 1000), dtype=tf.float64)
    y = tf.random.normal((1000, 1000), dtype=tf.float64)

    # Matrix multiplication
    z = x @ y

    print(z[0, 0])
```