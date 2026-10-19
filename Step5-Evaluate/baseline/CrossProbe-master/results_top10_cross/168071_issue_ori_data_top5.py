```python
import tensorflow as tf

# Create a tensor of zeros with shape (6, 0)
x0 = tf.zeros((6, 0))

# PyTorch's functional.pad takes padding in reverse order of dimensions (last dim first).
# For a 2D tensor, pad=(left, right, top, bottom).
# Here, pad=(0, 0, 0, 24) means:
#   - Last dimension (columns): left=0, right=24
#   - First dimension (rows): top=0, bottom=0
# TensorFlow's tf.pad takes paddings as [[top, bottom], [left, right]].
x1 = tf.pad(x0, [[0, 0], [0, 24]])

print(x1.shape)
```