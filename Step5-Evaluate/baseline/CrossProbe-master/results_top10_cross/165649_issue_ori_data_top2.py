```python
import tensorflow as tf

# Conversion: torch.full((2, 3), ...) -> tf.constant(..., shape=(2, 3), dtype=...)
# Conversion: torch.iinfo(torch.int64).min -> tf.int64.min
dividend = tf.constant(tf.int64.min, shape=(2, 3), dtype=tf.int64)
divisor = tf.constant(-1, shape=(3,), dtype=tf.int64)

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# Conversion: torch.fmod -> tf.math.fmod
result = tf.math.fmod(dividend, divisor)
print("Result:", result)
```