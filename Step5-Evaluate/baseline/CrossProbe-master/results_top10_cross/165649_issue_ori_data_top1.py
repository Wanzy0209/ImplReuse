```python
import tensorflow as tf

# Conversion: torch.iinfo(torch.int64).min is the minimum value for int64
min_int64 = -(2**63)

# Conversion: torch.full((2, 3), val, dtype=torch.int64, device='cpu')
# Note: TensorFlow defaults to CPU if no GPU is available or specified.
dividend = tf.constant(min_int64, shape=(2, 3), dtype=tf.int64)

# Conversion: torch.full((3,), -1, dtype=torch.int64, device='cpu')
divisor = tf.constant(-1, shape=(3,), dtype=tf.int64)

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

# Conversion: torch.fmod -> tf.math.fmod
result = tf.math.fmod(dividend, divisor)
print("Result:", result)
```