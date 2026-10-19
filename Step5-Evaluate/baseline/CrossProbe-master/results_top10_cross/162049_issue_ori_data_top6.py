```python
import tensorflow as tf

# Conversion: torch.nested.nested_tensor with layout=torch.jagged is equivalent to tf.ragged.constant
# Conversion: torch.arange is equivalent to tf.range
x = tf.ragged.constant(
    [tf.range(0, n) for n in (10, 20, 30)]
)

# Conversion: x.max(dim=1).values is equivalent to tf.reduce_max(x, axis=1)
print(tf.reduce_max(x, axis=1))
```