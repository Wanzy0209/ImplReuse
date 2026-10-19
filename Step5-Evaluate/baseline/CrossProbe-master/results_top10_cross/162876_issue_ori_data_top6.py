```python
import tensorflow as tf

# torch.aminmax equivalent
# Note: TensorFlow uses separate reduce_min and reduce_max operations
tf.reduce_min(tf.constant([1, -3, 5])), tf.reduce_max(tf.constant([1, -3, 5]))

# torch.return_types.aminmax equivalent
# Note: TensorFlow returns tensors directly, here represented as a tuple
(
    tf.constant(-3),
    tf.constant(5)
)
```