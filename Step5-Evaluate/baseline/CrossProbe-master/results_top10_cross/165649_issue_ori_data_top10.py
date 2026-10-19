```python
import tensorflow as tf
import numpy as np

# Conversion: torch.full((2, 3), val, dtype=torch.int64) -> tf.constant(val, shape=(2, 3), dtype=tf.int64)
# Conversion: torch.iinfo(torch.int64).min -> np.iinfo(np.int64).min
dividend = tf.constant(np.iinfo(np.int64).min, shape=(2, 3), dtype=tf.int64)
divisor = tf.constant(-1, shape=(3,), dtype=tf.int64)

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# Conversion: torch.fmod -> tf.math.fmod
result = tf.math.fmod(dividend, divisor)
print("Result:", result)
```