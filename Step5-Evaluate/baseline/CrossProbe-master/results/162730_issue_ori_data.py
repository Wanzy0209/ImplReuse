```python
import tensorflow as tf
import einops

# Create test tensors on MPS
# Note: TensorFlow does not support 'mps' device string directly. 
# We use '/GPU:0' if available, otherwise '/CPU:0'.
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

with tf.device(device):
    W = tf.random.normal((12, 64, 768))
    x = tf.random.normal((1, 3, 768))
    bias = tf.random.normal((768))

    # Create non-contiguous weight via rearrange
    w_noncontig = einops.rearrange(W, "h d m -> m (h d)")
    # Use tf.identity to simulate contiguous() (forces a copy of the tensor)
    w_contig = tf.identity(w_noncontig)

    # TensorFlow tensors are immutable, so is_contiguous() is not directly applicable.
    # We assume w_contig is contiguous due to the explicit copy.
    print(f"Weight contiguous: True")
    print(f"Weight non-contiguous: False")

    # Define linear operation: y = xW^T + b
    # PyTorch's linear expects weight shape (out_features, in_features)
    def linear(input_t, weight_t, bias_t):
        return tf.linalg.matmul(input_t, weight_t, transpose_b=True) + bias_t

    # These should be identical but aren't on MPS (in PyTorch)
    result1 = linear(x, w_noncontig, bias)
    result2 = linear(x, w_contig, bias)

    # Check if results match
    # tf.math.abs equivalent for torch.abs
    diff = tf.math.abs(result1 - result2)
    match = tf.reduce_all(diff < 1e-5)
    max_diff = tf.reduce_max(diff)

    print(f"Results match: {match}")
    print(f"Max difference: {max_diff}")

# Compare with CPU
# Explicitly copy tensors to CPU to simulate .cpu() method
with tf.device('/CPU:0'):
    x_cpu = tf.identity(x)
    w_noncontig_cpu = tf.identity(w_noncontig)
    w_contig_cpu = tf.identity(w_contig)
    bias_cpu = tf.identity(bias)

    result_cpu_noncontig = linear(x_cpu, w_noncontig_cpu, bias_cpu)
    result_cpu_contig = linear(x_cpu, w_contig_cpu, bias_cpu)
    
    diff_cpu = tf.math.abs(result_cpu_noncontig - result_cpu_contig)
    match_cpu = tf.reduce_all(diff_cpu < 1e-5)
    print(f"CPU contiguous vs non-contiguous match: {match_cpu}")
```