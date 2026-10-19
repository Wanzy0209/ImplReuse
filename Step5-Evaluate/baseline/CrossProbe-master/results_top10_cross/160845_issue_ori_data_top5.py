```python
import tensorflow as tf

# Set random seed
tf.random.set_seed(0)

# PyTorch "mps" is specific to Apple Silicon. 
# In TensorFlow, we use "/GPU:0" or "/CPU:0". 
# We default to GPU if available, otherwise CPU, to ensure the code runs.
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

shape = (400, )
# PyTorch: torch.arange(400, dtype=torch.long)
# TensorFlow: tf.range with dtype int64
idx = tf.range(400, dtype=tf.int64)
dim = 0

# Initialize t_mps on the specified device
with tf.device(device):
    t_mps = tf.zeros(shape, dtype=tf.complex64)

# Initialize t_cpu on CPU
with tf.device("/CPU:0"):
    t_cpu = tf.zeros(shape, dtype=tf.complex64)

trailing = shape[dim+1:]
# len(idx) works in eager mode for TF tensors
src_shape = (len(idx),) + trailing

# Create source tensors on the target device
with tf.device(device):
    # PyTorch: torch.randn
    src_imag = tf.random.normal(src_shape, dtype=tf.float32)
    # PyTorch: torch.zeros_like
    src_real = tf.zeros_like(src_imag)
    # PyTorch: torch.complex
    src = tf.complex(src_real, src_imag)

    # PyTorch: t_mps.index_add_(dim, idx.to(device), src)
    # TensorFlow: tf.tensor_scatter_nd_add
    # Indices must be shaped (N, rank) for scatter_nd operations
    indices = tf.expand_dims(idx, axis=-1)
    t_mps = tf.tensor_scatter_nd_add(t_mps, indices, src)

# Perform operation on CPU
# PyTorch: t_cpu.index_add_(dim, idx.cpu(), src.cpu())
with tf.device("/CPU:0"):
    # Ensure src and idx are available for CPU operation
    src_cpu = tf.identity(src)
    indices_cpu = tf.expand_dims(idx, axis=-1)
    t_cpu = tf.tensor_scatter_nd_add(t_cpu, indices_cpu, src_cpu)

# PyTorch: t_mps.imag.abs().sum().item()
# TensorFlow: tf.math.imag, tf.abs, tf.reduce_sum, .numpy()
print("MPS imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_mps))).numpy())
print("CPU imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_cpu))).numpy())

# PyTorch: (t_mps.cpu() - t_cpu).abs().max().item()
# Note: t_mps is copied to CPU implicitly or explicitly for the subtraction
diff = tf.abs(tf.identity(t_mps) - t_cpu)
print("max abs diff:", tf.reduce_max(diff).numpy())
```