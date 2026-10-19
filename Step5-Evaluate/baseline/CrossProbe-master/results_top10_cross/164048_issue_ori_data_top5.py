```python
import tensorflow as tf

# PyTorch: torch.randint(0, 20, (4, 87, 1056, 736), device="cuda")
# TensorFlow uses tf.random.uniform with dtype=int64 to match torch.randint
# and tf.device context for GPU placement
with tf.device('/GPU:0'):
    mask = tf.random.uniform((4, 87, 1056, 736), minval=0, maxval=20, dtype=tf.int64)
    to_apply = tf.constant([True, False, False, True])
    # PyTorch: mask[to_apply]
    # TensorFlow uses tf.boolean_mask for boolean indexing
    tf.boolean_mask(mask, to_apply)
```