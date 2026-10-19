```python
import tensorflow as tf

# PyTorch: th.nested.nested_tensor([th.ones(3, 2, 3), th.ones(4, 2, 3)], layout=th.jagged)
# TensorFlow: tf.ragged.stack creates a RaggedTensor from a list of tensors
x = tf.ragged.stack([tf.ones((3, 2, 3)), tf.ones((4, 2, 3))])

# PyTorch: th.cat([x, x])
# TensorFlow: tf.concat concatenates tensors along the specified axis
tf.concat([x, x], axis=0)
```