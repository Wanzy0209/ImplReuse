```python
import tensorflow as tf

# Conversion: torch.ones -> tf.ones
# Note: PyTorch's 'mps' device is specific to Apple Silicon. 
# TensorFlow creates tensors on the default device (CPU or GPU) unless explicitly placed in a context.
# The shape (1 << 31) + 5 evaluates to 2147483653.
a = tf.ones(shape=(2, (1 << 31) + 5), dtype=tf.int8)

# Conversion: PyTorch indexing a[1, -2] -> TensorFlow indexing a[1, -2]
print(a[1, -2])

# Conversion: PyTorch slicing a[:, -2] -> TensorFlow slicing a[:, -2]
print(a[:, -2])
```