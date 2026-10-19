```python
import tensorflow as tf

# PyTorch's torch.nested.nested_tensor is equivalent to tf.RaggedTensor for handling jagged/ragged arrays.
# torch.arange is equivalent to tf.range.
x = tf.ragged.constant(
    [tf.range(n) for n in (10, 20, 30)]
)

# PyTorch's x.max(dim=1).values computes the maximum along the inner dimension.
# In TensorFlow, this is done using tf.reduce_max with axis=1.
print(tf.reduce_max(x, axis=1))
```