```python
import tensorflow as tf

# Conversion: torch.rand -> tf.random.uniform
# Note: TensorFlow device placement is typically handled via context managers (e.g., with tf.device('/GPU:0'))
# rather than direct arguments in tensor constructors.
weight = tf.random.uniform((768, 768), dtype=tf.float32)
bias = tf.random.uniform((768,), dtype=tf.float32)

shape = (5, 499, 768)
stride = (0, 768, 1)
storage_offset = 0

# Calculation of the number of elements in the underlying storage
numel = storage_offset + sum((shape[i] - 1) * stride[i] for i in range(len(shape))) + 1

# Conversion: torch.arange -> tf.range
base = tf.range(numel, dtype=tf.float32)

# Conversion: torch.as_strided
# The stride (0, 768, 1) indicates that the first dimension is a broadcast (stride 0),
# while the remaining dimensions (499, 768) follow a contiguous layout (strides 768, 1).
# We reshape the base tensor to (499, 768) and broadcast it to (5, 499, 768).
# storage_offset is 0, so we start from the beginning of the tensor.
input = tf.broadcast_to(tf.reshape(base, (499, 768)), shape)

# Conversion: F.linear -> tf.linalg.matmul
# PyTorch's F.linear(input, weight, bias) computes input @ weight.T + bias.
# TensorFlow's matmul requires explicit transpose_b=True to match the behavior.
tf.linalg.matmul(input, weight, transpose_b=True) + bias
```