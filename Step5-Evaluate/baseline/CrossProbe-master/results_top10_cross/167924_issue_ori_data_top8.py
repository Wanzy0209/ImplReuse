```python
import tensorflow as tf

# Conversion: torch.tensor -> tf.constant
# Note: Device placement in TF is typically handled via context managers (e.g., with tf.device('/GPU:0'))
counts = tf.constant([0, 1, 0])
# Conversion: torch.arange -> tf.range
data = tf.range(2)
# Conversion: torch.repeat_interleave -> tf.repeat
tf.repeat(data, counts[1:3], axis=0)
```