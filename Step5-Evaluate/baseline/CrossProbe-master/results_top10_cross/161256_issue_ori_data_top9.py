```python
import tensorflow as tf

# Note: TensorFlow does not have a global set_default_dtype function.
# The dtype must be specified explicitly in the operation calls.
# We use tf.float64 to match torch.float64.

# Note: TensorFlow uses a context manager for device placement.
# '/GPU:0' is the equivalent of device="cuda".
with tf.device('/GPU:0'):
    # torch.randn generates random numbers from a standard normal distribution.
    # In TensorFlow, this is tf.random.normal.
    # We explicitly pass dtype=tf.float64 to match the source default.
    x = tf.random.normal((1000, 1000), dtype=tf.float64)
    y = tf.random.normal((1000, 1000), dtype=tf.float64)

    # The @ operator performs matrix multiplication in TensorFlow as well.
    z = x @ y

    print(z[0, 0])
```