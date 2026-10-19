```python
import tensorflow as tf

# Conversion: torch.randint(0, 20, ...) -> tf.random.uniform(..., dtype=tf.int32)
# Note: Device placement in TensorFlow is often implicit or handled via tf.device contexts.
mask = tf.random.uniform((4, 87, 1056, 736), minval=0, maxval=20, dtype=tf.int32)

# Conversion: torch.tensor([...]) -> tf.constant([...])
to_apply = tf.constant([True, False, False, True], dtype=tf.bool)

# Conversion: mask[to_apply] -> tf.boolean_mask(mask, to_apply, axis=0)
# TensorFlow uses tf.boolean_mask for boolean indexing along a specific axis.
tf.boolean_mask(mask, to_apply, axis=0)
```