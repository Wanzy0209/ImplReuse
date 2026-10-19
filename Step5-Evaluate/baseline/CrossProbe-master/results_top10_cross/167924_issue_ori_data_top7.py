```python
import tensorflow as tf

# torch.tensor -> tf.constant
# Note: device="mps" is PyTorch specific; tf.constant does not take a device argument.
# TensorFlow manages device placement automatically.
counts = tf.constant([0, 1, 0], dtype=tf.int64)

# torch.arange -> tf.range
data = tf.range(2, dtype=tf.int64)

# torch.repeat_interleave -> tf.repeat
# axis=0 corresponds to dim=0
tf.repeat(data, counts[1:3], axis=0)
```