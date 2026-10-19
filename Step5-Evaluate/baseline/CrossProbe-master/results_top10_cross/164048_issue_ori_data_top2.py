```python
import tensorflow as tf

# Conversion: torch.randint(0, 20, size, device="cuda") -> tf.random.uniform on GPU
# Note: Explicit device placement mimics device="cuda". dtype=int32 matches torch default.
with tf.device('/GPU:0'):
    mask = tf.random.uniform((4, 87, 1056, 736), minval=0, maxval=20, dtype=tf.int32)

    # Conversion: torch.tensor(data, device="cuda") -> tf.constant(data)
    to_apply = tf.constant([True, False, False, True])

    # Conversion: tensor[bool_mask] -> tf.boolean_mask(tensor, mask)
    tf.boolean_mask(mask, to_apply)
```