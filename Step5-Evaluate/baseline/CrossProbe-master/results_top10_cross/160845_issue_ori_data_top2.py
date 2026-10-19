```python
import tensorflow as tf

# torch.manual_seed(0)
tf.random.set_seed(0)

# device = "mps"
# In TensorFlow, "mps" (Metal Performance Shaders) is abstracted under the GPU device.
# We use '/GPU:0' as the equivalent target device.
physical_devices = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if physical_devices else '/CPU:0'

shape = (400, )
# idx = torch.arange(400, dtype=torch.long)
# torch.long is int64
idx = tf.range(400, dtype=tf.int64)
dim = 0

# t_mps = torch.zeros(shape, dtype=torch.complex64, device=device)
with tf.device(device):
    t_mps = tf.zeros(shape, dtype=tf.complex64)

# t_cpu = torch.zeros(shape, dtype=torch.complex64, device="cpu")
with tf.device('/CPU:0'):
    t_cpu = tf.zeros(shape, dtype=tf.complex64)

trailing = shape[dim+1:]
# src_shape = (len(idx),) + trailing
src_shape = (tf.shape(idx)[0],) + trailing

# src_imag = torch.randn(src_shape, dtype=torch.float32, device=device)
with tf.device(device):
    src_imag = tf.random.normal(src_shape, dtype=tf.float32)
    # src_real = torch.zeros_like(src_imag)
    src_real = tf.zeros_like(src_imag)
    # src = torch.complex(src_real, src_imag)
    src = tf.complex(src_real, src_imag)

# t_mps.index_add_(dim, idx.to(device), src)
# TensorFlow does not have in-place index_add. We use tf.tensor_scatter_nd_add.
# Indices must be reshaped to (N, rank) for scatter operations.
with tf.device(device):
    indices = tf.reshape(idx, (-1, 1))
    t_mps = tf.tensor_scatter_nd_add(t_mps, indices, src)

# t_cpu.index_add_(dim, idx.cpu(), src.cpu())
with tf.device('/CPU:0'):
    # Move src to CPU
    src_cpu = tf.identity(src)
    indices_cpu = tf.reshape(idx, (-1, 1))
    t_cpu = tf.tensor_scatter_nd_add(t_cpu, indices_cpu, src_cpu)

# print("MPS imag sum:", t_mps.imag.abs().sum().item())
print("MPS imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_mps))).numpy())

# print("CPU imag sum:", t_cpu.imag.abs().sum().item())
print("CPU imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_cpu))).numpy())

# print("max abs diff:", (t_mps.cpu() - t_cpu).abs().max().item())
# t_mps is on GPU, t_cpu is on CPU. TF handles the device transfer in the operation.
diff = tf.abs(t_mps - t_cpu)
print("max abs diff:", tf.reduce_max(diff).numpy())
```