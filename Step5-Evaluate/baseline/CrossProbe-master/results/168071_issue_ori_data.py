```python
import tensorflow as tf

# Create a tensor of zeros with shape (6, 0)
x0 = tf.zeros((6, 0))

# Pad the tensor
# PyTorch padding format is (left, right, top, bottom) for 2D tensors.
# TensorFlow padding format is [[top, bottom], [left, right]].
# Source padding (0, 0, 0, 24) maps to TF padding [[0, 24], [0, 0]].
x1 = tf.pad(x0, [[0, 24], [0, 0]])

print(x1.shape)
```