```python
import tensorflow as tf

# Note: Device placement in TensorFlow is typically handled via context managers (e.g., with tf.device('/GPU:0'))
# or runtime configuration, rather than as a constructor argument.
counts = tf.constant([0, 1, 0])
data = tf.range(2)
# tf.repeat is the TensorFlow equivalent of torch.repeat_interleave
tf.repeat(data, counts[1:3], axis=0)
```