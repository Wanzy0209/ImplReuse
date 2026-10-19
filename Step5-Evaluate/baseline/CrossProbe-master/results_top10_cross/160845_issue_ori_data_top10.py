```python
import tensorflow as tf

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(0)

# Conversion: PyTorch "mps" device maps to GPU in TensorFlow.
# We check for GPU availability to avoid errors on CPU-only machines.
gpus = tf.config.list_physical_devices('GPU')
device = "/GPU:0" if gpus else "/CPU:0"

shape = (400, )
# Conversion: torch.arange -> tf.range
idx = tf.range(400, dtype=tf.int64)
dim = 0

# Conversion: torch.zeros -> tf.zeros
with tf.device(device):
    t_mps = tf.zeros(shape, dtype=tf.complex64)

with tf.device("/CPU:0"):
    t_cpu = tf.zeros(shape, dtype=tf.complex64)

trailing = shape[dim+1:]
src_shape = (len(idx),) + trailing

# Conversion: torch.randn -> tf.random.normal
with tf.device(device):
    src_imag = tf.random.normal(src_shape, dtype=tf.float32)
    # Conversion: torch.zeros_like -> tf.zeros_like
    src_real = tf.zeros_like(src_imag)
    # Conversion: torch.complex -> tf.complex
    src = tf.complex(src_real, src_imag)

# Conversion: torch.index_add_ -> tf.tensor_scatter_nd_add
# Note: tf.tensor_scatter_nd_add requires indices to have shape (N, rank)
with tf.device(device):
    indices = tf.reshape(idx, (-1, 1))
    t_mps = tf.tensor_scatter_nd_add(t_mps, indices, src)

with tf.device("/CPU:0"):
    # Conversion: src.cpu() -> tf.identity(src) to ensure data is on CPU
    src_cpu = tf.identity(src)
    indices_cpu = tf.reshape(idx, (-1, 1))
    t_cpu = tf.tensor_scatter_nd_add(t_cpu, indices_cpu, src_cpu)

# Conversion: .imag, .abs(), .sum(), .item() -> tf.math.imag, tf.math.abs, tf.reduce_sum, .numpy()
print("MPS imag sum:", tf.reduce_sum(tf.math.abs(tf.math.imag(t_mps))).numpy())
print("CPU imag sum:", tf.reduce_sum(tf.math.abs(tf.math.imag(t_cpu))).numpy())
# Conversion: .cpu() is implicit in TF ops, .max() -> tf.reduce_max
print("max abs diff:", tf.reduce_max(tf.math.abs(t_mps - t_cpu)).numpy())
```