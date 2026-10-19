```python
import tensorflow as tf

# torch.randint(0, 20, size) -> tf.random.uniform(shape, minval=0, maxval=20, dtype=tf.int32)
# Note: Device placement is implicit in TF or managed via tf.device, not passed in constructor.
mask = tf.random.uniform((4, 87, 1056, 736), minval=0, maxval=20, dtype=tf.int32)

# torch.tensor(data) -> tf.constant(data)
to_apply = tf.constant([True, False, False, True])

# PyTorch boolean indexing mask[boolean] -> tf.boolean_mask(tensor, mask)
tf.boolean_mask(mask, to_apply)
```