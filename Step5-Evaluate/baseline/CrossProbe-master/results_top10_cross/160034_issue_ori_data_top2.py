```python
import tensorflow as tf

# PyTorch uses "mps" for Apple Silicon. TensorFlow uses standard device strings.
# On Mac with TensorFlow, the GPU is usually registered as '/GPU:0'.
device = "/GPU:0"  # Requires Apple Silicon Mac

# TF uses context managers for device placement instead of passing device to constructors
with tf.device(device):
    image = tf.zeros(10, dtype=tf.complex64)
    data = tf.ones(3, dtype=tf.complex64)
    indices = tf.constant([1, 3, 5])

# This fails on MPS with complex64
# TF equivalent of index_put_ with accumulate=True is tf.tensor_scatter_nd_add.
# Indices must be shaped (N, 1) for a 1D tensor.
# TF tensors are immutable, so we reassign the result.
image = tf.tensor_scatter_nd_add(image, tf.reshape(indices, [-1, 1]), data)
```