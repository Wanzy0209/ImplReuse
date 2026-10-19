```python
import tensorflow as tf

# Conversion: torch.ones -> tf.ones
# Note: PyTorch 'mps' device is specific to Apple Silicon; TF uses default device placement.
# Shape arguments are passed as a list in TensorFlow.
a = tf.ones([2, (1 << 31) + 5], dtype=tf.int8)

# Conversion: Slicing syntax is consistent between PyTorch and TensorFlow.
print(a[1, -2])
print(a[:, -2])
```