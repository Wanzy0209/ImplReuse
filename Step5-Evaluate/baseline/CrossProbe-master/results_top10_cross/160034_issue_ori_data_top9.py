```python
import tensorflow as tf

# device = torch.device("mps")  # Requires Apple Silicon Mac
# Conversion: TensorFlow uses context managers for device placement.
# '/GPU:0' is the standard equivalent for GPU execution.
with tf.device('/GPU:0'):  # Requires Apple Silicon Mac
    image = tf.zeros((10,), dtype=tf.complex64)
    data = tf.ones((3,), dtype=tf.complex64)
    indices = tf.constant([1, 3, 5])

    # This fails on MPS with complex64
    # Conversion: index_put_ with accumulate=True maps to tensor_scatter_nd_add.
    # TF tensors are immutable, so we reassign the result.
    # Indices must be shaped (num_updates, rank) -> (3, 1).
    image = tf.tensor_scatter_nd_add(image, tf.reshape(indices, (-1, 1)), data)
```