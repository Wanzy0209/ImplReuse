```python
import tensorflow as tf

# Conversion: torch.manual_seed(0)
tf.random.set_seed(0)

# Conversion: device = "mps"
# In TensorFlow, MPS (Metal Performance Shaders) is typically mapped to the default GPU.
# We use '/GPU:0' to represent the accelerator device.
device = "/GPU:0"

shape = (400, )
# Conversion: torch.arange(400, dtype=torch.long)
idx = tf.range(400, dtype=tf.int64)
dim = 0

# Conversion: torch.zeros(shape, dtype=torch.complex64, device=device)
with tf.device(device):
    t_mps = tf.zeros(shape, dtype=tf.complex64)

# Conversion: torch.zeros(shape, dtype=torch.complex64, device="cpu")
with tf.device("/CPU:0"):
    t_cpu = tf.zeros(shape, dtype=tf.complex64)

trailing = shape[dim+1:]
src_shape = (len(idx),) + trailing

# Conversion: torch.randn(src_shape, dtype=torch.float32, device=device)
# Note: Original code generates src on 'device' (MPS)
with tf.device(device):
    src_imag = tf.random.normal(src_shape, dtype=tf.float32)
    # Conversion: torch.zeros_like(src_imag)
    src_real = tf.zeros_like(src_imag)
    # Conversion: torch.complex(src_real, src_imag)
    src = tf.complex(src_real, src_imag)

# Conversion: t_mps.index_add_(dim, idx.to(device), src)
# PyTorch index_add_ is in-place. TF uses tensor_scatter_nd_add.
# Indices for tensor_scatter_nd_add must be shaped (N, rank).
with tf.device(device):
    indices = tf.reshape(idx, [-1, 1])
    t_mps = tf.tensor_scatter_nd_add(t_mps, indices, src)

# Conversion: t_cpu.index_add_(dim, idx.cpu(), src.cpu())
# Original code moves src and idx to CPU.
with tf.device("/CPU:0"):
    # src.cpu() and idx.cpu() imply copying data to CPU
    src_cpu = tf.identity(src)
    idx_cpu = tf.identity(idx)
    
    indices_cpu = tf.reshape(idx_cpu, [-1, 1])
    t_cpu = tf.tensor_scatter_nd_add(t_cpu, indices_cpu, src_cpu)

# Conversion: t_mps.imag.abs().sum().item()
# .imag is tf.math.imag, .abs is tf.abs, .sum is tf.reduce_sum, .item() is .numpy()
print("MPS imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_mps))).numpy())

# Conversion: t_cpu.imag.abs().sum().item()
print("CPU imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_cpu))).numpy())

# Conversion: (t_mps.cpu() - t_cpu).abs().max().item()
# t_mps is on GPU, t_cpu is on CPU. TF handles cross-device ops automatically.
print("max abs diff:", tf.reduce_max(tf.abs(t_mps - t_cpu)).numpy())
```