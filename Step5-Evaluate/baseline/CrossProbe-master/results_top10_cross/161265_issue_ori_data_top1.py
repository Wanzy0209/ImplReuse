```python
import tensorflow as tf

# Conversion: torch.ones -> tf.ones
# Note: PyTorch's 'mps' device is specific to Apple Silicon. 
# TensorFlow handles device placement automatically or via tf.device context.
# Note: The shape (2, (1 << 31) + 5) is very large (~4GB).
a = tf.ones((2, (1 << 31) + 5), dtype=tf.int8)

# Conversion: PyTorch indexing is directly supported in TensorFlow
print(a[1, -2])
print(a[:, -2])
```