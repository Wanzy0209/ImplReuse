```python
import tensorflow as tf

# Conversion: torch.nested.nested_tensor with jagged layout maps to tf.RaggedTensor.
# We use tf.ragged.constant to construct it from a list of varying-length tensors.
x = tf.ragged.constant(
    [tf.range(0, n) for n in (10, 20, 30)]
)
# Conversion: x.max(dim=1).values computes the max along the ragged dimension.
# TensorFlow's reduce_max returns values directly (indices are not returned by default).
print(tf.reduce_max(x, axis=1))
```