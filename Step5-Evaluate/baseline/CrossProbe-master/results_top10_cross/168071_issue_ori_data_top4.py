```python
import tensorflow as tf

# PyTorch: pt.zeros((6, 0))
x0 = tf.zeros((6, 0))

# PyTorch: pt.nn.functional.pad(x0, (0, 0, 0, 24))
# Note: PyTorch padding order is (left, right, top, bottom) for 2D tensors.
# TensorFlow padding order is [[top, bottom], [left, right]].
# Therefore, (0, 0, 0, 24) becomes [[0, 24], [0, 0]].
x1 = tf.pad(x0, [[0, 24], [0, 0]])

print(x1.shape)
```