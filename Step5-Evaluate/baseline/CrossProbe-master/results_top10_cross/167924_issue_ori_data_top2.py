```python
import tensorflow as tf

# Note: device='mps' is PyTorch specific; TF uses default device (CPU/GPU)
counts = tf.constant([0, 1, 0], dtype=tf.int64)
data = tf.range(2, dtype=tf.int64)
# tf.repeat is the equivalent of torch.repeat_interleave
tf.repeat(data, counts[1:3], axis=0)
```