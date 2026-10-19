```python
import tensorflow as tf
from collections import namedtuple

# Define a named tuple to mimic torch.return_types.aminmax
# PyTorch returns a named tuple, so we replicate this structure for compatibility
AminMaxResult = namedtuple('AminMaxResult', ['min', 'max'])

# torch.aminmax computes both min and max.
# TensorFlow uses separate ops (reduce_min, reduce_max), so we combine them into the named tuple.
AminMaxResult(
    min=tf.reduce_min(tf.constant([1, -3, 5])),
    max=tf.reduce_max(tf.constant([1, -3, 5]))
)

# Replicate the explicit construction of the return type object
# torch.tensor -> tf.constant
AminMaxResult(
    min=tf.constant(-3),
    max=tf.constant(5)
)
```