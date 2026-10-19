```python
import tensorflow as tf
import einops

# Create test tensors
# Note: TensorFlow handles device placement automatically (e.g., GPU/MPS on Mac)
W = tf.random.normal((12, 64, 768))
x = tf.random.normal((1, 3, 768))
bias = tf.random.normal((768,))

# Create non-contiguous weight via rearrange
w_noncontig = einops.rearrange(W, "h d m -> m (h d)")
# Use tf.identity to ensure a distinct tensor, mimicking .contiguous() behavior
w_contig = tf.identity(w_noncontig)

# TensorFlow does not expose memory contiguity checks like PyTorch.
# Memory layout is managed internally by the TensorFlow runtime.
print(f"Weight contiguous: True (Managed internally)")
print(f"Weight non-contiguous: False (Managed internally)")

# PyTorch linear: x @ weight.T + bias
# TensorFlow equivalent: tf.linalg.matmul(x, weight, transpose_b=True) + bias
result1 = tf.linalg.matmul(x, w_noncontig, transpose_b=True) + bias
result2 = tf.linalg.matmul(x, w_contig, transpose_b=True) + bias

# Check if results match using TensorFlow operations
# tf.reduce_all returns a scalar tensor, .numpy() extracts the value for printing
match = tf.reduce_all(tf.abs(result1 - result2) <= 1e-5)
print(f"Results match: {match.numpy()}")
print(f"Max difference: {tf.reduce_max(tf.abs(result1 - result2)).numpy()}")

# Compare with CPU
# Explicitly scope operations to CPU device
with tf.device('/CPU:0'):
    result_cpu_noncontig = tf.linalg.matmul(x, w_noncontig, transpose_b=True) + bias
    result_cpu_contig = tf.linalg.matmul(x, w_contig, transpose_b=True) + bias
    cpu_match = tf.reduce_all(tf.abs(result_cpu_noncontig - result_cpu_contig) <= 1e-5)
    print(f"CPU contiguous vs non-contiguous match: {cpu_match.numpy()}")
```