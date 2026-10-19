```python
import tensorflow as tf

# PyTorch: [torch.arange(0, n) for n in (10, 20, 30)]
# TensorFlow: Create a list of 1D tensors using tf.range
tensors = [tf.range(n) for n in (10, 20, 30)]

# PyTorch: torch.nested.nested_tensor(..., layout=torch.jagged)
# TensorFlow: tf.RaggedTensor is the equivalent for nested/jagged tensors.
# We construct it from the list of tensors using row lengths.
flat_values = tf.concat(tensors, axis=0)
row_lengths = [len(t) for t in tensors]
x = tf.RaggedTensor.from_row_lengths(flat_values, row_lengths)

# PyTorch: x.max(dim=1).values
# TensorFlow: tf.reduce_max computes the max along the specified axis.
# For RaggedTensors, axis=1 corresponds to the inner (jagged) dimension.
print(tf.reduce_max(x, axis=1))
```