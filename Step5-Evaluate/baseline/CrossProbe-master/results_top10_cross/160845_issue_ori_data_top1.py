```python
import tensorflow as tf

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(0)
# Note: TensorFlow handles device placement implicitly or via tf.device context.
# We preserve the variable name for structure, but explicit MPS placement is not standard in TF scripts.
device = "mps"

shape = (400, )
# Conversion: torch.arange -> tf.range, dtype=torch.long -> dtype=tf.int64
idx = tf.range(400, dtype=tf.int64)
dim = 0

# Conversion: torch.zeros -> tf.zeros
t_mps = tf.zeros(shape, dtype=tf.complex64)
t_cpu = tf.zeros(shape, dtype=tf.complex64)

trailing = shape[dim+1:]
# Conversion: len(idx) -> tf.shape(idx)[0] to ensure compatibility with graph tensors
src_shape = (tf.shape(idx)[0],) + trailing

# Conversion: torch.randn -> tf.random.normal
src_imag = tf.random.normal(src_shape, dtype=tf.float32)
# Conversion: torch.zeros_like -> tf.zeros_like
src_real = tf.zeros_like(src_imag)
# Conversion: torch.complex -> tf.complex
src = tf.complex(src_real, src_imag)

# Conversion: torch.index_add_ -> tf.tensor_scatter_nd_add
# PyTorch's index_add_ is in-place, while TF returns a new tensor.
# TF requires indices to be shaped (N, rank) for scatter operations.
indices = tf.reshape(idx, (-1, 1))

t_mps = tf.tensor_scatter_nd_add(t_mps, indices, src)
t_cpu = tf.tensor_scatter_nd_add(t_cpu, indices, src)

# Conversion: .imag -> tf.math.imag, .abs() -> tf.abs, .sum() -> tf.reduce_sum, .item() -> .numpy()
print("MPS imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_mps))).numpy())
print("CPU imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_cpu))).numpy())

# Conversion: .cpu() is implicit in TF operations, .max() -> tf.reduce_max
print("max abs diff:", tf.reduce_max(tf.abs(t_mps - t_cpu)).numpy())
```