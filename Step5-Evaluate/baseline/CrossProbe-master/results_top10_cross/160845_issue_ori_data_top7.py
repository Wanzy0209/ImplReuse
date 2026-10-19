```python
import tensorflow as tf

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(0)

# Conversion: "mps" device maps to GPU in TensorFlow
device = "/GPU:0"

shape = (400, )
# Conversion: torch.arange -> tf.range, torch.long -> tf.int64
idx = tf.range(400, dtype=tf.int64)
dim = 0

# Conversion: torch.zeros with device placement
with tf.device(device):
    t_mps = tf.zeros(shape, dtype=tf.complex64)

with tf.device("/CPU:0"):
    t_cpu = tf.zeros(shape, dtype=tf.complex64)

trailing = shape[dim+1:]
# Conversion: len(idx) works on TF tensors in eager mode
src_shape = (len(idx),) + trailing

# Create source tensors on the target device
with tf.device(device):
    # Conversion: torch.randn -> tf.random.normal
    src_imag = tf.random.normal(src_shape, dtype=tf.float32)
    # Conversion: torch.zeros_like -> tf.zeros_like
    src_real = tf.zeros_like(src_imag)
    # Conversion: torch.complex -> tf.complex
    src = tf.complex(src_real, src_imag)

# Conversion: torch.index_add_ -> tf.tensor_scatter_nd_add
# Note: TF scatter_nd requires indices to have shape (N, rank)
idx_mps = tf.reshape(idx, [-1, 1])
with tf.device(device):
    t_mps = tf.tensor_scatter_nd_add(t_mps, idx_mps, src)

# Conversion: index_add on CPU
# Explicitly moving tensors to CPU context to match .cpu() calls
with tf.device("/CPU:0"):
    src_cpu = tf.identity(src)
    idx_cpu = tf.reshape(idx, [-1, 1])
    t_cpu = tf.tensor_scatter_nd_add(t_cpu, idx_cpu, src_cpu)

# Conversion: .imag -> tf.math.imag, .abs() -> tf.abs, .sum() -> tf.reduce_sum, .item() -> .numpy()
print("MPS imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_mps))).numpy())
print("CPU imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_cpu))).numpy())

# Conversion: t_mps.cpu() -> tf.identity(t_mps) on CPU device
with tf.device("/CPU:0"):
    t_mps_cpu = tf.identity(t_mps)
    diff = tf.abs(t_mps_cpu - t_cpu)
    print("max abs diff:", tf.reduce_max(diff).numpy())
```