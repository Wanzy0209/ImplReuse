```python
import tensorflow as tf

# Create a zero tensor with shape (6, 0)
x0 = tf.zeros((6, 0))

# Pad the tensor.
# PyTorch padding format: (left, right, top, bottom) -> (0, 0, 0, 24)
# TensorFlow padding format: [[top, bottom], [left, right]] -> [[0, 24], [0, 0]]
x1 = tf.pad(x0, [[0, 24], [0, 0]])

print(x1.shape)
```