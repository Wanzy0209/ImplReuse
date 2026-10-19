```python
import tensorflow as tf

# Create a tensor of zeros with shape (6, 0)
x0 = tf.zeros((6, 0))

# Pad the tensor.
# PyTorch padding (0, 0, 0, 24) applies to the last 2 dimensions in reverse order (left, right, top, bottom).
# For input shape (6, 0):
# - Last dimension (dim 1): padding (0, 24) -> size becomes 0 + 0 + 24 = 24
# - Second to last dimension (dim 0): padding (0, 0) -> size remains 6
# TensorFlow paddings format: [[dim_0_start, dim_0_end], [dim_1_start, dim_1_end]]
x1 = tf.pad(x0, [[0, 0], [0, 24]])

print(x1.shape)
```