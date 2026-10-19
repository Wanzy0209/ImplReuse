```python
import tensorflow as tf

# Create a tensor of zeros with shape (6, 0)
x0 = tf.zeros((6, 0))

# PyTorch: pt.nn.functional.pad(x0, (0, 0, 0, 24))
# PyTorch padding order starts from the last dimension: (left, right, top, bottom)
# TensorFlow padding order is: [[top, bottom], [left, right]]
# Mapping (0, 0, 0, 24) -> Dim 1: (0, 0), Dim 0: (0, 24)
x1 = tf.pad(x0, [[0, 24], [0, 0]])

print(x1.shape)
```