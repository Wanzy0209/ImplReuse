```python
import tensorflow as tf

# Conversion: torch.ones -> tf.ones
# Note: 'mps' device is PyTorch specific; TensorFlow uses the default available device.
# Shape: (1 << 31) + 5 = 2147483653
a = tf.ones([2, (1 << 31) + 5], dtype=tf.int8)

# Conversion: Slicing syntax is identical in TensorFlow
# Note: In eager execution, print() outputs the Tensor object, not the raw value directly.
print(a[1, -2])
print(a[:, -2])
```