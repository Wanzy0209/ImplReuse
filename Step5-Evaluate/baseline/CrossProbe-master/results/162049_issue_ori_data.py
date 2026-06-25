```python
import tensorflow as tf

# Convert torch.nested.nested_tensor to tf.ragged.constant
# Convert torch.arange to tf.range
# Note: torch.arange defaults to int64, so we specify dtype=tf.int64
x = tf.ragged.constant(
    [tf.range(0, n, dtype=tf.int64) for n in (10, 20, 30)]
)

# Convert x.max(dim=1).values to tf.reduce_max(x, axis=1)
# PyTorch's max returns a named tuple (values, indices), while tf.reduce_max returns just the values
print(tf.reduce_max(x, axis=1))
```