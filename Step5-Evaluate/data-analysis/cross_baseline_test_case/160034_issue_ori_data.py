```python
import tensorflow as tf

# PyTorch uses "mps" for Apple Silicon. TensorFlow uses "/GPU:0" or "/CPU:0".
# Note: TF device placement is typically handled via context managers.
device = "/GPU:0"  # Requires Apple Silicon Mac

with tf.device(device):
    # tf.zeros takes a shape tuple or integer
    image = tf.zeros(10, dtype=tf.complex64)
    data = tf.ones(3, dtype=tf.complex64)
    indices = tf.constant([1, 3, 5])

    # This fails on MPS with complex64
    # Conversion: index_put_ with accumulate=True maps to tf.tensor_scatter_nd_add.
    # TF requires indices to be rank-2 (N, 1) for 1D updates.
    # TF tensors are immutable, so we reassign the result.
    image = tf.tensor_scatter_nd_add(image, tf.expand_dims(indices, axis=1), data)
```