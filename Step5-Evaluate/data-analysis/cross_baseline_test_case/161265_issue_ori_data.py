```python
import tensorflow as tf

# Note: PyTorch's 'mps' device is specific to Apple Silicon. 
# TensorFlow handles device placement via context managers (e.g., with tf.device('/GPU:0')).
# The shape (2, (1 << 31) + 5) is very large (~4.3GB for int8).
a = tf.ones((2, (1 << 31) + 5), dtype=tf.int8)

print(a[1, -2])
print(a[:, -2])
```