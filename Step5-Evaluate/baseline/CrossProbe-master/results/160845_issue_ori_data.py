```python
import tensorflow as tf

# Conversion: torch.manual_seed(0)
tf.random.set_seed(0)

# Conversion: device = "mps" -> TensorFlow uses '/GPU:0' for GPU acceleration
device = "/GPU:0"

shape = (400, )
# Conversion: torch.arange(400, dtype=torch.long)
idx = tf.range(400, dtype=tf.int64)
dim = 0

# Conversion: torch.zeros(..., device=device)
with tf.device(device):
    t_mps = tf.zeros(shape, dtype=tf.complex64)

# Conversion: torch.zeros(..., device="cpu")
with tf.device("/CPU:0"):
    t_cpu = tf.zeros(shape, dtype=tf.complex64)

trailing = shape[dim+1:]
src_shape = (len(idx),) + trailing

# Conversion: torch.randn(..., device=device)
with tf.device(device):
    src_imag = tf.random.normal(src_shape, dtype=tf.float32)
    src_real = tf.zeros_like(src_imag)
    src = tf.complex(src_real, src_imag)

# Conversion: torch.index_add_(dim, idx, src)
# TF requires indices to be shaped (N, rank) for scatter operations
indices = tf.reshape(idx, (-1, 1))

with tf.device(device):
    # t_mps.index_add_(dim, idx.to(device), src)
    t_mps = tf.tensor_scatter_nd_add(t_mps, indices, src)

# Conversion: src.cpu() and t_cpu.index_add_
with tf.device("/CPU:0"):
    # Move src to CPU
    src_cpu_imag = tf.identity(src_imag)
    src_cpu_real = tf.zeros_like(src_cpu_imag)
    src_cpu = tf.complex(src_cpu_real, src_cpu_imag)

    # t_cpu.index_add_(dim, idx.cpu(), src.cpu())
    t_cpu = tf.tensor_scatter_nd_add(t_cpu, indices, src_cpu)

# Conversion: Print statements
# t_mps.imag -> tf.math.imag
# .abs() -> tf.abs
# .sum() -> tf.reduce_sum
# .item() -> .numpy()
print("MPS imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_mps))).numpy())
print("CPU imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_cpu))).numpy())
print("max abs diff:", tf.reduce_max(tf.abs(t_mps - t_cpu)).numpy())
```