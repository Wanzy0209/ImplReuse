```python
import tensorflow as tf

# PyTorch: torch.device("mps")
# TensorFlow: Device placement is handled via context manager. 
# '/GPU:0' is the standard equivalent for the primary accelerator (MPS on Mac via metal).
device = '/GPU:0'  # Requires Apple Silicon Mac

# PyTorch: device=device argument in constructor
# TensorFlow: Use tf.device context manager for placement.
# Note: tf.Variable is required for in-place updates (index_put_ equivalent).
with tf.device(device):
    image = tf.Variable(tf.zeros(10, dtype=tf.complex64))
    data = tf.ones(3, dtype=tf.complex64)
    indices = tf.constant([1, 3, 5])

# PyTorch: image.index_put_(indices=(indices,), values=data, accumulate=True)
# TensorFlow: tf.tensor_scatter_nd_add performs the accumulate operation (add values at indices).
# Indices must be reshaped to (N, rank) -> (3, 1) for a 1D tensor.
# This fails on MPS with complex64
image.assign(tf.tensor_scatter_nd_add(image, tf.reshape(indices, (-1, 1)), data))
```