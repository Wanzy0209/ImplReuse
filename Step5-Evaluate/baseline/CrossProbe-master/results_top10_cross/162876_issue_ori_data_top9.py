```python
import tensorflow as tf
from collections import namedtuple

# Conversion: Define a namedtuple to mimic torch.return_types.aminmax structure
aminmax = namedtuple('aminmax', ['min', 'max'])

# Conversion: torch.aminmax computes both min and max.
# TensorFlow requires separate calls to tf.reduce_min and tf.reduce_max.
# We wrap them in the namedtuple to match the return structure.
aminmax(
    min=tf.reduce_min(tf.constant([1, -3, 5])),
    max=tf.reduce_max(tf.constant([1, -3, 5]))
)

# Conversion: torch.return_types.aminmax construction
aminmax(
    min=tf.constant(-3),
    max=tf.constant(5)
)
```