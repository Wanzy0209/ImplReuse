```python
import tensorflow as tf

# torch.set_default_dtype sets the default floating point dtype.
# TensorFlow does not have a global default dtype setter, so we specify it explicitly.
dtype = tf.float64

# device="cuda" places tensors on the GPU. In TensorFlow, we use a device context manager.
with tf.device('/GPU:0'):
    # torch.randn generates samples from a standard normal distribution.
    # We use tf.random.normal with the specified dtype.
    x = tf.random.normal((1000, 1000), dtype=dtype)
    y = tf.random.normal((1000, 1000), dtype=dtype)

    # Matrix multiplication operator @ is supported in TensorFlow 2.x
    z = x @ y

    print(z[0, 0])
```