```python
import tensorflow as tf

# Conversion: torch.randint(0, 20, ...) -> tf.random.uniform(..., minval=0, maxval=20, dtype=tf.int32)
# Conversion: device="cuda" -> tf.device('/GPU:0') context manager
with tf.device('/GPU:0'):
    mask = tf.random.uniform((4, 87, 1056, 736), minval=0, maxval=20, dtype=tf.int32)
    to_apply = tf.constant([True, False, False, True], dtype=tf.bool)

# Conversion: mask[to_apply] -> tf.boolean_mask(mask, to_apply)
tf.boolean_mask(mask, to_apply)
```