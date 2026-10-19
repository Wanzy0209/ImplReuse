```python
import tensorflow as tf

# Note: TensorFlow handles device placement explicitly (e.g., with tf.device) 
# rather than as a constructor argument, so device="mps" is omitted.
counts = tf.constant([0, 1, 0])
data = tf.range(2)
# tf.repeat is the TensorFlow equivalent of torch.repeat_interleave
tf.repeat(data, counts[1:3], axis=0)
```