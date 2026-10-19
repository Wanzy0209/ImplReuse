```python
import tensorflow as tf

# torch._inductor.config.combo_kernels = True
# Note: PyTorch inductor config has no direct equivalent in TensorFlow.

@tf.function
def fn(x, y, z):
    # Conversion: x.sum(1) -> tf.reduce_sum(x, axis=1)
    # Conversion: y.mean(1) -> tf.reduce_mean(y, axis=1)
    # Conversion: z.cumsum(1) -> tf.math.cumsum(z, axis=1)
    return tf.reduce_sum(x, axis=1), tf.reduce_mean(y, axis=1), tf.math.cumsum(z, axis=1)

# Conversion: torch.rand -> tf.random.uniform
# Note: Device placement in TF is implicit or handled via context managers.
# Using '/GPU:0' to match 'device="cuda"'.
with tf.device('/GPU:0'):
    inps = (
        tf.random.uniform((16, 128)),
        tf.random.uniform((32, 128)),
        tf.random.uniform((32, 256)),
    )

fn(*inps)
```