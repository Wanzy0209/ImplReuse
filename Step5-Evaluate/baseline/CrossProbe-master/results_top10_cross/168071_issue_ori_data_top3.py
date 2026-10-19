```python
import tensorflow as tf

# Create a tensor of zeros with shape (6, 0)
x0 = tf.zeros((6, 0))

# PyTorch pad format: (left, right, top, bottom) = (0, 0, 0, 24)
# TensorFlow pad format: [[top, bottom], [left, right]]
# Conversion: [[0, 24], [0, 0]]
x1 = tf.pad(x0, [[0, 24], [0, 0]])

print(x1.shape)
```