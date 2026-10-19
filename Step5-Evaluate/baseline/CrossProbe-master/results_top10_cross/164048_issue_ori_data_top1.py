```python
import tensorflow as tf

# torch.randint(0, 20, (4, 87, 1056, 736), device="cuda")
# In TensorFlow 2.x, operations are placed on GPU automatically if available.
# tf.random.uniform generates floats by default, so dtype is set to int32/64.
mask = tf.random.uniform((4, 87, 1056, 736), minval=0, maxval=20, dtype=tf.int32)

# torch.tensor([True, False, False, True], device="cuda")
to_apply = tf.constant([True, False, False, True], dtype=tf.bool)

# mask[to_apply] -> Boolean indexing
# PyTorch supports direct boolean indexing, while TensorFlow uses tf.boolean_mask.
selected_mask = tf.boolean_mask(mask, to_apply)
```