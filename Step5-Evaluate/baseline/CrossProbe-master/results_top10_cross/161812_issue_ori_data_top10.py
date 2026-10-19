```python
import tensorflow as tf

# PyTorch: th.nested.nested_tensor([th.ones(3, 2, 3), th.ones(4, 2, 3)], layout=th.jagged)
# TensorFlow uses RaggedTensor to represent nested/jagged data.
# tf.ragged.constant constructs a RaggedTensor from a nested list or list of tensors.
x = tf.ragged.constant([tf.ones((3, 2, 3)), tf.ones((4, 2, 3))])

# PyTorch: th.cat([x, x])
# TensorFlow uses tf.concat to concatenate tensors, including RaggedTensors.
# Concatenating along axis 0 joins the outer dimension (the list of tensors).
tf.concat([x, x], axis=0)
```