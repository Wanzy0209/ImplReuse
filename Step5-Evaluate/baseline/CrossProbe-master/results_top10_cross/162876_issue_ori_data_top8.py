```python
import tensorflow as tf
from collections import namedtuple

# Define a namedtuple to mimic torch.return_types.aminmax
# as TensorFlow does not have a direct equivalent return type.
AminMax = namedtuple('AminMax', ['min', 'max'])

# Helper function to mimic torch.aminmax
def aminmax(input_tensor):
    return AminMax(min=tf.reduce_min(input_tensor), max=tf.reduce_max(input_tensor))

# torch.tensor([1, -3, 5]) -> tf.constant([1, -3, 5])
aminmax(tf.constant([1, -3, 5]))

# torch.return_types.aminmax(...) -> AminMax(...)
# tensor(...) -> tf.constant(...)
AminMax(
    min=tf.constant(-3),
    max=tf.constant(5)
)
```