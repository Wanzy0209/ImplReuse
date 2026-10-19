```python
import tensorflow as tf
from collections import namedtuple

# Define a namedtuple to mimic torch.return_types.aminmax structure
AminMax = namedtuple('AminMax', ['min', 'max'])

# torch.aminmax computes min and max. TF uses separate ops.
# We combine them into a tuple to mimic the single call structure.
tf.math.reduce_min(tf.constant([1, -3, 5])), tf.math.reduce_max(tf.constant([1, -3, 5]))

# torch.return_types.aminmax is a specific PyTorch type.
# Using the defined namedtuple to construct the equivalent object.
AminMax(min=tf.constant(-3), max=tf.constant(5))
```