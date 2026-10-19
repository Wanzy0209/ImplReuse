```python
import tensorflow as tf

# PyTorch's empty_strided allows custom memory layout (strides).
# TensorFlow manages memory layout internally and does not support
# arbitrary strides for new tensor allocations. We replicate the shape.
input_tensor = tf.empty([1792, 1899, 160])

# torch.rand_like creates a random tensor with the same shape and dtype.
# In TensorFlow, we use tf.random.uniform.
output_tensor = tf.random.uniform(input_tensor.shape, dtype=input_tensor.dtype)

# PyTorch's rand_like preserves the strides of the input.
# TensorFlow tensors are contiguous and do not expose a .stride() method.
# We verify the shape is preserved instead.
# assert input_tensor.stride() == output_tensor.stride()
assert input_tensor.shape == output_tensor.shape
```