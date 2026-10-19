```python
import tensorflow as tf

# Conversion: torch.nested.nested_tensor with layout=torch.jagged maps to tf.RaggedTensor.
# torch.arange is converted to Python range for the list comprehension passed to tf.ragged.constant.
x = tf.ragged.constant(
    [list(range(n)) for n in (10, 20, 30)],
    dtype=tf.int64, # Match torch.arange default dtype (int64)
)

# Conversion: x.max(dim=1).values computes the maximum along the jagged dimension.
# In TensorFlow, this is done via reduce_max(axis=1).
print(x.reduce_max(axis=1))
```