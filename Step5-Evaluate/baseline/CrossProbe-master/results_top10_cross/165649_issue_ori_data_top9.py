```python
import tensorflow as tf
import numpy as np

# Conversion: torch.full with torch.iinfo -> tf.constant with np.iinfo
# torch.int64 maps to tf.int64 (np.int64)
dividend = tf.constant(np.iinfo(np.int64).min, dtype=tf.int64, shape=(2, 3))
# Conversion: torch.full -> tf.constant
divisor = tf.constant(-1, dtype=tf.int64, shape=(3,))

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# Conversion: torch.fmod -> tf.math.fmod
result = tf.math.fmod(dividend, divisor)
print("Result:", result)
```