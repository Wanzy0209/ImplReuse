```python
import tensorflow as tf

# Note: 'mps' device is PyTorch specific; using default device in TensorFlow
counts = tf.constant([0, 1, 0])
data = tf.range(2)
# PyTorch's repeat_interleave maps to TensorFlow's repeat
tf.repeat(data, counts[1:3], axis=0)
```