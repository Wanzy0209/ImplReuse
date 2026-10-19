```python
import tensorflow as tf

# PyTorch: torch.aminmax(torch.tensor([1, -3, 5]))
# TensorFlow: tf.math.reduce_min and tf.math.reduce_max
# Note: TensorFlow does not have a single function for both, so we compute them separately and return as a tuple
(tf.math.reduce_min(tf.constant([1, -3, 5])), tf.math.reduce_max(tf.constant([1, -3, 5])))

# PyTorch: torch.return_types.aminmax(min=tensor(-3), max=tensor(5))
# TensorFlow: Constructing a tuple of tensors as there is no specific return type class
(
    tf.constant(-3),
    tf.constant(5)
)
```