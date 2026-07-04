```python
import tensorflow as tf
import numpy as np

# Conversion: torch.iinfo(torch.int64).min -> np.iinfo(np.int64).min
# Conversion: torch.full -> tf.fill
# Note: device='cpu' is default in TensorFlow
dividend = tf.fill((2, 3), tf.cast(np.iinfo(np.int64).min, tf.int64))
divisor = tf.fill((3,), tf.cast(-1, tf.int64))

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# Conversion: torch.fmod -> tf.math.truncatemod
result = tf.math.truncatemod(dividend, divisor)
print("Result:", result)
```