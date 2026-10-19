```python
import tensorflow as tf

# torch.randint(low, high, size) -> tf.random.uniform(shape, minval, maxval, dtype)
# Note: TensorFlow uses dtype=tf.int32 for integers, default is float32.
# Device placement in TF is typically handled via 'with tf.device(...):' context managers.
mask = tf.random.uniform((4, 87, 1056, 736), minval=0, maxval=20, dtype=tf.int32)

# torch.tensor(data, device) -> tf.constant(data, dtype)
to_apply = tf.constant([True, False, False, True], dtype=tf.bool)

# PyTorch boolean indexing tensor[mask] -> tf.boolean_mask(tensor, mask)
tf.boolean_mask(mask, to_apply)
```