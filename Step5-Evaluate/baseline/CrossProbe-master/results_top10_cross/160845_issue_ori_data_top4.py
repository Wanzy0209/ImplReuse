```python
import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(0)

# Define device. PyTorch uses "mps" (Metal Performance Shaders).
# In TensorFlow, the equivalent is typically "/GPU:0" if a GPU is available.
device = "/GPU:0"

shape = (400, )
# Create indices tensor
idx = tf.range(400, dtype=tf.int64)
dim = 0

# --- Operations on GPU (MPS equivalent) ---
with tf.device(device):
    t_mps = tf.zeros(shape, dtype=tf.complex64)
    
    # Calculate source shape
    trailing = shape[dim+1:]
    # tf.shape(idx)[0] is used to get the dynamic size, similar to len(idx)
    src_shape = (tf.shape(idx)[0],) + trailing
    
    # Generate random data on the specific device
    src_imag = tf.random.normal(src_shape, dtype=tf.float32)
    src_real = tf.zeros_like(src_imag)
    src = tf.complex(src_real, src_imag)
    
    # Perform index_add operation
    # tf.tensor_scatter_nd_add is the TensorFlow equivalent of index_add_
    # Indices must be shaped (N, rank) for scatter_nd operations
    indices = tf.expand_dims(idx, axis=1)
    t_mps = tf.tensor_scatter_nd_add(t_mps, indices, src)

# --- Operations on CPU ---
with tf.device("/CPU:0"):
    t_cpu = tf.zeros(shape, dtype=tf.complex64)
    
    # Recalculate source shape for CPU context
    src_shape_cpu = (tf.shape(idx)[0],) + trailing
    
    # Generate random data on CPU
    # Note: This generates different random numbers than the GPU call,
    # matching the behavior of the separate PyTorch calls.
    src_imag_cpu = tf.random.normal(src_shape_cpu, dtype=tf.float32)
    src_real_cpu = tf.zeros_like(src_imag_cpu)
    src_cpu = tf.complex(src_real_cpu, src_imag_cpu)
    
    # Perform index_add operation on CPU
    indices_cpu = tf.expand_dims(idx, axis=1)
    t_cpu = tf.tensor_scatter_nd_add(t_cpu, indices_cpu, src_cpu)

# --- Comparison ---
# Move t_mps to CPU for comparison, similar to t_mps.cpu() in PyTorch
with tf.device("/CPU:0"):
    t_mps_on_cpu = tf.identity(t_mps)

# Calculate metrics
# .imag -> tf.math.imag
# .abs() -> tf.abs
# .sum() -> tf.reduce_sum
# .item() -> .numpy()
print("MPS imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_mps_on_cpu))).numpy())
print("CPU imag sum:", tf.reduce_sum(tf.abs(tf.math.imag(t_cpu))).numpy())

# Calculate max absolute difference
diff = tf.abs(t_mps_on_cpu - t_cpu)
print("max abs diff:", tf.reduce_max(diff).numpy())
```