```python
import tensorflow as tf

# Conversion: torch.randint(0, 20, (4, 87, 1056, 736), device="cuda")
# Note: PyTorch default dtype for randint is int64. Device placement is automatic in TensorFlow.
mask = tf.random.uniform((4, 87, 1056, 736), minval=0, maxval=20, dtype=tf.int64)

# Conversion: torch.tensor([True, False, False, True], device="cuda")
to_apply = tf.constant([True, False, False, True])

# Conversion: mask[to_apply] -> tf.boolean_mask(mask, to_apply, axis=0)
# TensorFlow does not support boolean indexing via [] syntax directly.
tf.boolean_mask(mask, to_apply, axis=0)
```