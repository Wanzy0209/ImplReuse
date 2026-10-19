```python
import tensorflow as tf

# Conversion: TensorFlow requires explicit dtype specification instead of a global default.
# Original: torch.set_default_dtype(torch.float64)

# Conversion: Use tf.device context to place tensors on GPU.
# Original: x = torch.randn(..., device="cuda")
with tf.device('/GPU:0'):
    # Conversion: torch.randn -> tf.random.normal
    x = tf.random.normal((1000, 1000), dtype=tf.float64)
    y = tf.random.normal((1000, 1000), dtype=tf.float64)

z = x @ y
print(z[0, 0])
```