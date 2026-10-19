```python
import tensorflow as tf

# Conversion: torch.manual_seed(0) -> tf.random.set_seed(0)
tf.random.set_seed(0)
# Conversion: "mps" is specific to PyTorch/Mac. Using "/gpu:0" as the equivalent accelerator.
device = "/gpu:0"

shape = (400, )
# Conversion: torch.arange -> tf.range
idx = tf.range(400, dtype=tf.int64)
dim = 0

# Conversion: torch.zeros -> tf.zeros
with tf.device(device):
    t_mps = tf.zeros(shape, dtype=tf.complex64)

with tf.device("/cpu:0"):
    t_cpu = tf.zeros(shape, dtype=tf.complex64)

trailing = shape[dim+1:]
# Conversion: len(idx) -> tf.shape(idx)[0]
src_shape = (tf.shape(idx)[0],) + trailing

with tf.device(device):
    # Conversion: torch.randn -> tf.random.normal
    src_imag = tf.random.normal(src_shape, dtype=tf.float32)
    src_real = tf.zeros_like(src_imag)
    # Conversion: torch.complex -> tf.complex
    src = tf.complex(src_real, src_imag)

# Conversion: torch.index_add_ -> tf.tensor_scatter_nd_add
# PyTorch: t_mps.index_add_(dim, idx.to(device), src)
# Indices for scatter_nd_add must be (N, rank)
indices_mps = tf.expand_dims(idx, axis=-1)
# Ensure idx is on the correct device (mimicking .to(device))
with tf.device(device):
    # Note: tf.tensor_scatter_nd_add is not in-place, returns a new tensor
    t_mps = tf.tensor_scatter_nd_add(t_mps, indices_mps, src)

# PyTorch: t_cpu.index_add_(dim, idx.cpu(), src.cpu())
# Move src to CPU
with tf.device("/cpu:0"):
    src_cpu = tf.identity(src)
    indices_cpu = tf.expand_dims(idx, axis=-1)
    t_cpu = tf.tensor_scatter_nd_add(t_cpu, indices_cpu, src_cpu)

# Conversion: .imag -> tf.math.imag, .abs() -> tf.abs, .sum() -> tf.reduce_sum
# Conversion: .item() -> .numpy().item()
print("MPS imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_mps))).numpy().item())
print("CPU imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_cpu))).numpy().item())
# Conversion: t_mps.cpu() -> tf.identity(t_mps) (explicit copy to CPU context if needed, though ops handle it)
# We compute diff. t_mps is on GPU, t_cpu on CPU. TF handles this.
diff = tf.abs(tf.identity(t_mps) - t_cpu)
print("max abs diff:", tf.reduce_max(diff).numpy().item())
```