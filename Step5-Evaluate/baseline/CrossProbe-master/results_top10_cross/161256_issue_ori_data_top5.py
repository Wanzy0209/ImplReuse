```python
import tensorflow as tf

# torch.set_default_dtype(torch.float64)
# TensorFlow does not have a global default dtype setting.
# We specify dtype=tf.float64 explicitly in the operations below.

# device="cuda" is handled via tf.device context manager
with tf.device('/GPU:0'):
    # x = torch.randn(1000, 1000, device="cuda")
    x = tf.random.normal((1000, 1000), dtype=tf.float64)
    # y = torch.randn(1000, 1000, device="cuda")
    y = tf.random.normal((1000, 1000), dtype=tf.float64)

    # z = x @ y
    z = x @ y

print(z[0, 0])
```