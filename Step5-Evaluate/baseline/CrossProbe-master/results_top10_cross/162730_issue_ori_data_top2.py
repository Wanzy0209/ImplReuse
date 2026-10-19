```python
import tensorflow as tf
import einops

# Create test tensors on GPU (MPS equivalent)
device = '/GPU:0'
with tf.device(device):
    W = tf.random.normal((12, 64, 768))
    x = tf.random.normal((1, 3, 768))
    bias = tf.random.normal((768,))

# Create non-contiguous weight via rearrange
# Note: In TensorFlow, tensors are immutable and layout is handled internally.
# We use tf.identity to mimic the explicit copy of .contiguous().
w_noncontig = einops.rearrange(W, "h d m -> m (h d)")
w_contig = tf.identity(w_noncontig)

# PyTorch specific check: is_contiguous() is not available in TensorFlow.
# TF operations generally handle non-contiguous memory layouts transparently.
print(f"Weight contiguous: True (TF implicit)")
print(f"Weight non-contiguous: True (TF implicit)")

# These should be identical but aren't on MPS (in PyTorch)
# PyTorch linear: x @ w.T + b
# TensorFlow equivalent: tf.linalg.matmul(x, w, transpose_b=True) + bias
# Using tf.nn.bias_add as per context for bias addition
result1 = tf.nn.bias_add(tf.linalg.matmul(x, w_noncontig, transpose_b=True), bias)
result2 = tf.nn.bias_add(tf.linalg.matmul(x, w_contig, transpose_b=True), bias)

# torch.allclose(a, b, atol=1e-5)
match = tf.reduce_all(tf.abs(result1 - result2) < 1e-5)
print(f"Results match: {match}")

# torch.abs(...).max()
max_diff = tf.reduce_max(tf.abs(result1 - result2))
print(f"Max difference: {max_diff}")

# Compare with CPU (works correctly)
# Explicitly moving tensors to CPU context
with tf.device('/CPU:0'):
    x_cpu = tf.identity(x)
    w_noncontig_cpu = tf.identity(w_noncontig)
    w_contig_cpu = tf.identity(w_contig)
    bias_cpu = tf.identity(bias)

    result_cpu_noncontig = tf.nn.bias_add(tf.linalg.matmul(x_cpu, w_noncontig_cpu, transpose_b=True), bias_cpu)
    result_cpu_contig = tf.nn.bias_add(tf.linalg.matmul(x_cpu, w_contig_cpu, transpose_b=True), bias_cpu)
    
    match_cpu = tf.reduce_all(tf.abs(result_cpu_noncontig - result_cpu_contig) < 1e-5)
    print(f"CPU contiguous vs non-contiguous match: {match_cpu}")
```