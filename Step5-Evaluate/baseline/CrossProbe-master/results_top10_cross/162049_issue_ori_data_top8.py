```python
import tensorflow as tf

# PyTorch's nested.nested_tensor with jagged layout is equivalent to TensorFlow's RaggedTensor.
# We use tf.ragged.constant to construct the ragged tensor from a list of tensors.
x = tf.ragged.constant(
    [tf.range(0, n) for n in (10, 20, 30)]
)

# PyTorch's x.max(dim=1).values computes the maximum along the jagged dimension (axis 1).
# TensorFlow's tf.reduce_max performs the equivalent reduction operation.
print(tf.reduce_max(x, axis=1))
```