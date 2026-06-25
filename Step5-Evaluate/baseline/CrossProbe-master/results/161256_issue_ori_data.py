```python
import tensorflow as tf

# PyTorch: torch.set_default_dtype(torch.float64)
# TensorFlow does not have a global default dtype setter that affects all tensor creation.
# We explicitly pass dtype=tf.float64 to the random generation functions below.
# We also use a context manager to place tensors on the GPU.

with tf.device('/GPU:0'):
    # PyTorch: x = torch.randn(1000, 1000, device="cuda")
    x = tf.random.normal((1000, 1000), dtype=tf.float64)

    # PyTorch: y = torch.randn(1000, 1000, device="cuda")
    y = tf.random.normal((1000, 1000), dtype=tf.float64)

    # PyTorch: z = x @ y
    z = x @ y

print(z[0, 0])
```