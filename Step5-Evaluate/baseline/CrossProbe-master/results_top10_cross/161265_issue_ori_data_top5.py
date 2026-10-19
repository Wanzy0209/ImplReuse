```python
import tensorflow as tf

# Conversion: torch.ones -> tf.ones
# Conversion: device='mps' -> tf.device('/GPU:0') (Standard TF GPU placement)
# Note: The dimension (1 << 31) + 5 exceeds int32 max, handled as int64 in TF.
with tf.device('/GPU:0'):
    a = tf.ones([2, (1 << 31) + 5], dtype=tf.int8)

# Conversion: Slicing is supported natively in TF 2.x
print(a[1, -2])
print(a[:, -2])
```