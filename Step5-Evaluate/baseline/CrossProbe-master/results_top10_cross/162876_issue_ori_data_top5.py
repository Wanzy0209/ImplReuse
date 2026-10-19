```python
import tensorflow as tf
from collections import namedtuple

# Conversion: Define a named tuple to mimic torch.return_types.aminmax
AminMaxResult = namedtuple('AminMaxResult', ['min', 'max'])

# Conversion: Define a helper to mimic torch.aminmax
def aminmax(input_tensor):
    return AminMaxResult(min=tf.math.reduce_min(input_tensor),
                         max=tf.math.reduce_max(input_tensor))

# Conversion: torch.tensor -> tf.constant
# Conversion: torch.aminmax -> custom aminmax helper
aminmax(tf.constant([1, -3, 5]))

# Conversion: torch.return_types.aminmax -> AminMaxResult
# Conversion: tensor -> tf.constant
AminMaxResult(min=tf.constant(-3), max=tf.constant(5))
```