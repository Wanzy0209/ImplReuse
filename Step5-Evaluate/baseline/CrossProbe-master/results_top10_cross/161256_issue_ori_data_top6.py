```python
import tensorflow as tf

# Conversion: torch.set_default_dtype(torch.float64) -> tf.keras.backend.set_floatx('float64')
tf.keras.backend.set_floatx('float64')

# Conversion: torch.randn(..., device="cuda") -> tf.random.normal inside tf.device context
# Note: Explicit dtype=tf.float64 is required for tf.random.normal to respect the global setting
with tf.device('/GPU:0'):
    x = tf.random.normal((1000, 1000), dtype=tf.float64)
    y = tf.random.normal((1000, 1000), dtype=tf.float64)

z = x @ y
# Conversion: print tensor value -> print(tensor.numpy())
print(z[0, 0].numpy())
```