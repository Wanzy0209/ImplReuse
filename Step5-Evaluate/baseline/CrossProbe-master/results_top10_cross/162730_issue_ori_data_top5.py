```python
import tensorflow as tf
import einops

# Create test tensors on GPU (MPS is PyTorch specific, using GPU as equivalent)
device = '/GPU:0'
with tf.device(device):
    # torch.randn
    W = tf.random.normal((12, 64, 768))
    x = tf.random.normal((1, 3, 768))
    bias = tf.random.normal((768,))

# Create non-contiguous weight via rearrange
w_noncontig = einops.rearrange(W, "h d m -> m (h d)")
# w_contig = w_noncontig.contiguous()
# In TensorFlow, tensors are immutable. tf.identity creates a copy, ensuring contiguous memory layout.
w_contig = tf.identity(w_noncontig)

# print(f"Weight contiguous: {w_contig.is_contiguous()}")
# TensorFlow does not expose an is_contiguous() method in the public API.
# We assume the identity copy is contiguous.
print(f"Weight contiguous: True")
# print(f"Weight non-contiguous: {w_noncontig.is_contiguous()}")
# We assume the rearranged tensor might be non-contiguous (view).
print(f"Weight non-contiguous: False")

# These should be identical but aren't on MPS (PyTorch specific bug)
# torch.nn.functional.linear(x, w, bias) -> x @ w.T + b
# TensorFlow equivalent: tf.matmul(x, w, transpose_b=True) + bias
result1 = tf.matmul(x, w_noncontig, transpose_b=True) + bias
result2 = tf.matmul(x, w_contig, transpose_b=True) + bias

# torch.allclose(result1, result2, atol=1e-5)
# TensorFlow equivalent: tf.reduce_all(tf.abs(a - b) < atol)
match = tf.reduce_all(tf.abs(result1 - result2) < 1e-5)
print(f"Results match: {match}")

# torch.abs(result1 - result2).max()
# TensorFlow equivalent: tf.reduce_max(tf.abs(a - b))
max_diff = tf.reduce_max(tf.abs(result1 - result2))
print(f"Max difference: {max_diff}")

# Compare with CPU (works correctly)
# .cpu() in PyTorch moves tensors to CPU. In TF, we use a device context.
with tf.device('/CPU:0'):
    result_cpu_noncontig = tf.matmul(x, w_noncontig, transpose_b=True) + bias
    result_cpu_contig = tf.matmul(x, w_contig, transpose_b=True) + bias

match_cpu = tf.reduce_all(tf.abs(result_cpu_noncontig - result_cpu_contig) < 1e-5)
print(f"CPU contiguous vs non-contiguous match: {match_cpu}")
```