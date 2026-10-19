```python
import tensorflow as tf
from collections import namedtuple

# Conversion: Define a named tuple to mimic torch.return_types.aminmax
AminMaxResult = namedtuple('AminMaxResult', ['min', 'max'])

# Conversion: torch.aminmax(torch.tensor([1, -3, 5]))
# TensorFlow uses tf.reduce_min and tf.reduce_max
input_tensor = tf.constant([1, -3, 5])
min_val = tf.reduce_min(input_tensor)
max_val = tf.reduce_max(input_tensor)
AminMaxResult(min=min_val, max=max_val)

# Conversion: torch.return_types.aminmax(min=tensor(-3), max=tensor(5))
AminMaxResult(
    min=tf.constant(-3),
    max=tf.constant(5)
)
```