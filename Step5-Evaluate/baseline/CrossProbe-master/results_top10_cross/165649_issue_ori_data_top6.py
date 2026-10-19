```python
import tensorflow as tf
import numpy as np

# Conversion: torch.iinfo(torch.int64).min -> np.iinfo(np.int64).min
# Conversion: torch.full -> tf.constant with shape argument
dividend = tf.constant(np.iinfo(np.int64).min, shape=(2, 3), dtype=tf.int64)
divisor = tf.constant(-1, shape=(3,), dtype=tf.int64)

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# Conversion: torch.fmod -> tf.math.fmod
result = tf.math.fmod(dividend, divisor)
print("Result:", result)
```