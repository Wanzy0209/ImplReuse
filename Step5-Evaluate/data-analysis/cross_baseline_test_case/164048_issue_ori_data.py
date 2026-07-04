```python
import tensorflow as tf

# PyTorch uses device="cuda", TensorFlow uses device context '/GPU:0'
with tf.device('/GPU:0'):
    # torch.randint(low, high, size) -> tf.random.uniform(shape, minval, maxval, dtype)
    # Note: PyTorch default dtype for randint is int64
    mask = tf.random.uniform((4, 87, 1056, 736), minval=0, maxval=20, dtype=tf.int64)

    # torch.tensor(data) -> tf.constant(data)
    to_apply = tf.constant([True, False, False, True])

    # PyTorch boolean indexing mask[to_apply] -> tf.boolean_mask(mask, to_apply)
    tf.boolean_mask(mask, to_apply)
```