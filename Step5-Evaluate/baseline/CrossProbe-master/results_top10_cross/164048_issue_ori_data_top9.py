```python
import tensorflow as tf

# Conversion: torch.randint(0, 20, ...) -> tf.random.uniform(..., dtype=tf.int64)
# Note: PyTorch defaults to int64 for randint, TF defaults to float32.
# Device placement is implicit in TensorFlow.
mask = tf.random.uniform((4, 87, 1056, 736), minval=0, maxval=20, dtype=tf.int64)

# Conversion: torch.tensor(...) -> tf.constant(...)
to_apply = tf.constant([True, False, False, True])

# Conversion: mask[to_apply] -> tf.boolean_mask(mask, to_apply)
# TensorFlow does not support direct boolean indexing with [].
tf.boolean_mask(mask, to_apply)
```