```python
import tensorflow as tf

# torch.tensor -> tf.constant
# Note: Device placement is handled implicitly by TensorFlow
counts = tf.constant([0, 1, 0])

# torch.arange -> tf.range
data = tf.range(2)

# torch.repeat_interleave -> tf.repeat
# Repeats elements of data along axis 0 based on the counts tensor slice
tf.repeat(data, counts[1:3], axis=0)
```