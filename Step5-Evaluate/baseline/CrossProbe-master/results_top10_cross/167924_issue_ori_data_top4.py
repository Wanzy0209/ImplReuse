```python
import tensorflow as tf

# Note: 'mps' device is PyTorch specific. In TensorFlow, device placement is handled
# via context managers (e.g., with tf.device('/gpu:0')). We create tensors on the default device here.
counts = tf.constant([0, 1, 0])
data = tf.range(2)
# tf.repeat is the TensorFlow equivalent of torch.repeat_interleave
tf.repeat(data, counts[1:3], axis=0)
```