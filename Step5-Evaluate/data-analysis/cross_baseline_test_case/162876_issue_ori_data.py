```python
import tensorflow as tf
from collections import namedtuple

# Conversion: Define a namedtuple to mimic torch.return_types.aminmax
aminmax = namedtuple('aminmax', ['min', 'max'])

# Conversion: torch.aminmax computes min and max of the tensor
# TensorFlow does not have a direct equivalent, so we compute min and max separately
input_tensor = tf.constant([1, -3, 5])
aminmax(min=tf.math.reduce_min(input_tensor), max=tf.math.reduce_max(input_tensor))

# Conversion: torch.return_types.aminmax(...) instantiates the result object
aminmax(
    min=tf.constant(-3),
    max=tf.constant(5)
)
```