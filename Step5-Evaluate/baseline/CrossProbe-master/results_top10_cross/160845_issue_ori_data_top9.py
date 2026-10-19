```python
import tensorflow as tf

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(0)

# Conversion: PyTorch "mps" device -> TensorFlow "/device:GPU:0"
# Note: TensorFlow uses standard device strings. "mps" is specific to PyTorch/Metal.
device = "/device:GPU:0" if tf.config.list_physical_devices('GPU') else "/device:CPU:0"

shape = (400, )
# Conversion: torch.arange -> tf.range
idx = tf.range(400, dtype=tf.int64)
dim = 0

# Conversion: torch.zeros -> tf.zeros
with tf.device(device):
    t_mps = tf.zeros(shape, dtype=tf.complex64)

with tf.device("/device:CPU:0"):
    t_cpu = tf.zeros(shape, dtype=tf.complex64)

trailing = shape[dim+1:]
# Conversion: len(idx) -> tf.shape(idx)[0]
src_shape = (tf.shape(idx)[0],) + trailing

with tf.device(device):
    # Conversion: torch.randn -> tf.random.normal
    src_imag = tf.random.normal(src_shape, dtype=tf.float32)
    # Conversion: torch.zeros_like -> tf.zeros_like
    src_real = tf.zeros_like(src_imag)
    # Conversion: torch.complex -> tf.complex
    src = tf.complex(src_real, src_imag)

    # Conversion: torch.index_add_ -> tf.tensor_scatter_nd_add
    # PyTorch index_add_ is in-place. TF tensors are immutable, so we reassign.
    # Indices for scatter_nd_add must be shaped (N, rank).
    indices = tf.reshape(idx, [-1, 1])
    t_mps = tf.tensor_scatter_nd_add(t_mps, indices, src)

# CPU path
with tf.device("/device:CPU:0"):
    # Ensure src is on CPU for the CPU operation
    src_cpu = tf.identity(src)
    indices_cpu = tf.reshape(idx, [-1, 1])
    t_cpu = tf.tensor_scatter_nd_add(t_cpu, indices_cpu, src_cpu)

# Conversion: .imag -> tf.math.imag, .abs() -> tf.abs, .sum() -> tf.reduce_sum, .item() -> .numpy()
print("MPS imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_mps))).numpy())
print("CPU imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_cpu))).numpy())

# Conversion: .cpu() -> implicit handling in TF ops, .max() -> tf.reduce_max
# t_mps is on GPU, t_cpu is on CPU. TF handles the device transfer for the operation.
diff = tf.abs(t_mps - t_cpu)
print("max abs diff:", tf.reduce_max(diff).numpy())
```