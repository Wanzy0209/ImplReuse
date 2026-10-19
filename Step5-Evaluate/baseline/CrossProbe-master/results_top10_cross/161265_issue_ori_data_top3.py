```python
import tensorflow as tf

# Conversion: torch.ones -> tf.ones
# Note: PyTorch 'mps' device is specific to Apple Silicon. 
# TensorFlow device placement is handled via context managers (e.g., with tf.device('/GPU:0')) 
# and is not specified directly in the tensor constructor.
# torch.int8 maps to tf.int8.
a = tf.ones((2, (1 << 31) + 5), dtype=tf.int8)

# Slicing syntax is identical in TensorFlow
print(a[1, -2])
print(a[:, -2])
```