```python
import tensorflow as tf

# Set random seed for reproducibility
# PyTorch: torch.manual_seed(0)
tf.random.set_seed(0)

# PyTorch uses "mps" for Apple Silicon GPU.
# TensorFlow uses standard device strings like "/CPU:0" or "/GPU:0".
# We use "/CPU:0" here to ensure the code runs on all environments.
device = "/CPU:0"

shape = (400, )
# PyTorch: torch.arange(400, dtype=torch.long)
# TensorFlow: tf.range(400, dtype=tf.int64)
idx = tf.range(400, dtype=tf.int64)
dim = 0

# Initialize tensors on specific devices
# PyTorch: torch.zeros(..., device=device)
with tf.device(device):
    t_mps = tf.zeros(shape, dtype=tf.complex64)
with tf.device("/CPU:0"):
    t_cpu = tf.zeros(shape, dtype=tf.complex64)

trailing = shape[dim+1:]
# PyTorch: len(idx) works on the size of the tensor.
# TensorFlow: tf.shape(idx)[0] returns the dynamic shape size.
src_shape = (tf.shape(idx)[0],) + trailing

# Generate random source data on 'device'
with tf.device(device):
    # PyTorch: torch.randn
    src_imag = tf.random.normal(src_shape, dtype=tf.float32)
    # PyTorch: torch.zeros_like
    src_real = tf.zeros_like(src_imag)
    # PyTorch: torch.complex
    src = tf.complex(src_real, src_imag)

# PyTorch: t_mps.index_add_(dim, idx.to(device), src)
# TensorFlow: tf.tensor_scatter_nd_add
# indices must be shaped (N, rank) for scatter_nd_add
indices = tf.reshape(idx, (-1, 1))
t_mps = tf.tensor_scatter_nd_add(t_mps, indices, src)

# PyTorch: t_cpu.index_add_(dim, idx.cpu(), src.cpu())
# Move tensors to CPU explicitly to match source logic
with tf.device("/CPU:0"):
    src_cpu = tf.identity(src)
    idx_cpu = tf.identity(idx)
    indices_cpu = tf.reshape(idx_cpu, (-1, 1))
    t_cpu = tf.tensor_scatter_nd_add(t_cpu, indices_cpu, src_cpu)

# PyTorch: t_mps.imag.abs().sum().item()
# TensorFlow: tf.reduce_sum(tf.abs(tf.math.imag(t_mps))).numpy()
print("MPS imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_mps))).numpy())
print("CPU imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_cpu))).numpy())

# PyTorch: (t_mps.cpu() - t_cpu).abs().max().item()
# TensorFlow: tf.reduce_max(tf.abs(t_mps - t_cpu)).numpy()
print("max abs diff:", tf.reduce_max(tf.abs(t_mps - t_cpu)).numpy())
```