```python
import tensorflow as tf

x0 = tf.zeros((6, 0))
# PyTorch pad format: (left, right, top, bottom) for 2D tensors
# TensorFlow pad format: [[top, bottom], [left, right]]
x1 = tf.pad(x0, [[0, 24], [0, 0]])
print(x1.shape)
```