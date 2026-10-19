```python
import tensorflow as tf

# Note: TensorFlow does not support custom strides for tensor allocation like PyTorch's empty_strided.
# We use the shape [1792, 1899, 160] for the equivalent operation.
input_tensor = tf.empty(shape=[1792, 1899, 160])

# torch.rand_like creates a random tensor with the same shape and dtype
output_tensor = tf.random.uniform(shape=input_tensor.shape, dtype=input_tensor.dtype)

# TensorFlow does not expose .stride() in the public API.
# We check shape equality instead, which is the primary structural property in TF.
assert input_tensor.shape == output_tensor.shape
```