```python
import tensorflow as tf

# Create a RaggedTensor to handle jagged/nested data
# Equivalent to torch.nested.nested_tensor(..., layout=torch.jagged)
# tf.range(n) is equivalent to torch.arange(0, n)
x = tf.ragged.constant(
    [tf.range(n) for n in (10, 20, 30)]
)

# Compute max along the ragged dimension (axis 1)
# Equivalent to x.max(dim=1).values
print(tf.reduce_max(x, axis=1))
```