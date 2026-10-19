```python
import tensorflow as tf

# Conversion: torch.randint(low, high, size, device) -> tf.random.uniform(shape, minval, maxval, dtype)
# Note: PyTorch defaults to int64. In TensorFlow, device placement is handled by the runtime or tf.device context.
mask = tf.random.uniform((4, 87, 1056, 736), minval=0, maxval=20, dtype=tf.int64)

# Conversion: torch.tensor -> tf.constant
to_apply = tf.constant([True, False, False, True])

# Conversion: Boolean indexing -> tf.boolean_mask
tf.boolean_mask(mask, to_apply)
```