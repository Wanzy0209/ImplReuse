```python
import tensorflow as tf

# Conversion: torch.tensor -> tf.constant
# Note: device="mps" is specific to PyTorch for Apple Silicon.
# In TensorFlow, device placement is typically implicit or managed via tf.device.
counts = tf.constant([0, 1, 0])
data = tf.range(2)
# Conversion: torch.repeat_interleave -> tf.repeat
tf.repeat(data, counts[1:3], axis=0)
```