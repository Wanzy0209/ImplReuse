```python
import tensorflow as tf

# TensorFlow does not have a global set_default_dtype function.
# We specify dtype=tf.float64 explicitly in tensor creation to match torch.float64.
# torch.set_default_dtype(torch.float64)

# Use tf.device context to place tensors on GPU, equivalent to device="cuda"
with tf.device('/GPU:0'):
    x = tf.random.normal((1000, 1000), dtype=tf.float64)
    y = tf.random.normal((1000, 1000), dtype=tf.float64)

z = x @ y
print(z[0, 0])
```