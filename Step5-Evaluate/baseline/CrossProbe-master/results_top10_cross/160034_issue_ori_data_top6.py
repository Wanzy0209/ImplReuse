```python
import tensorflow as tf

# Conversion: PyTorch "mps" device maps to "/GPU:0" in TensorFlow (requires tensorflow-metal plugin)
device = "/GPU:0"

# Conversion: Device placement in TF is handled via context managers
with tf.device(device):
    # Conversion: torch.zeros -> tf.Variable(tf.zeros(...))
    # Using Variable to mimic the mutable behavior of PyTorch tensors for index_put_
    image = tf.Variable(tf.zeros(10, dtype=tf.complex64))
    
    # Conversion: torch.ones -> tf.ones
    data = tf.ones(3, dtype=tf.complex64)
    
    # Conversion: torch.tensor -> tf.constant
    indices = tf.constant([1, 3, 5], dtype=tf.int32)

# This fails on MPS with complex64
# Conversion: index_put_ with accumulate=True -> tensor_scatter_nd_add
# Note: indices must be expanded to shape (N, 1) for 1D tensor scatter
image.assign(tf.tensor_scatter_nd_add(image, tf.expand_dims(indices, axis=1), data))
```