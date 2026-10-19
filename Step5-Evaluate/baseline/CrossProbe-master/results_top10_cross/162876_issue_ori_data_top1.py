```python
import tensorflow as tf

# torch.aminmax computes both min and max. TF uses separate ops.
# Note: TF does not have a direct aminmax function.
tf.reduce_min(tf.constant([1, -3, 5])), tf.reduce_max(tf.constant([1, -3, 5]))

# torch.return_types.aminmax is a named tuple. TF returns tensors directly.
# We represent the result as a tuple here.
(tf.constant(-3), tf.constant(5))
```