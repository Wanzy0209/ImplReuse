```python
import tensorflow as tf

# Note: PyTorch's device="mps" (Apple Silicon) is handled via context managers in TensorFlow.
# e.g., with tf.device('/GPU:0'): ... or with tf.device('/CPU:0'): ...
counts = tf.constant([0, 1, 0])
data = tf.range(2)

# PyTorch's repeat_interleave is equivalent to tf.repeat in TensorFlow.
# counts[1:3] extracts the slice [1, 0].
# data [0, 1] repeated [1, 0] times results in [0].
tf.repeat(data, counts[1:3], axis=0)
```