```python
import tensorflow as tf
import einops

# Create test tensors on GPU (MPS equivalent)
# TensorFlow uses '/GPU:0' for the default GPU device.
device = '/GPU:0'
with tf.device(device):
    W = tf.random.normal((12, 64, 768))
    x = tf.random.normal((1, 3, 768))
    bias = tf.random.normal((768,))

    # Create non-contiguous weight via rearrange
    w_noncontig = einops.rearrange(W, "h d m -> m (h d)")
    # In TensorFlow, tensors are immutable. tf.identity creates a new tensor,
    # mimicking the .contiguous() call which ensures a copy in PyTorch.
    w_contig = tf.identity(w_noncontig)

    # TensorFlow tensors do not have an is_contiguous() property as memory layout 
    # is abstracted by the runtime. We preserve the print structure with a placeholder.
    print(f"Weight contiguous: True (N/A in TF)")
    print(f"Weight non-contiguous: True (N/A in TF)")

    # torch.nn.functional.linear(x, weight, bias) performs x @ weight.T + bias
    result1 = tf.linalg.matmul(x, w_noncontig, transpose_b=True) + bias
    result2 = tf.linalg.matmul(x, w_contig, transpose_b=True) + bias

    # Check if results match
    # tf.reduce_all returns a boolean tensor indicating if all elements satisfy the condition
    match = tf.reduce_all(tf.abs(result1 - result2) < 1e-5)
    print(f"Results match: {match.numpy()}")
    max_diff = tf.reduce_max(tf.abs(result1 - result2))
    print(f"Max difference: {max_diff.numpy()}")

# Compare with CPU (works correctly)
# We use tf.identity inside a CPU device block to move data from GPU to CPU
with tf.device('/CPU:0'):
    x_cpu = tf.identity(x)
    w_noncontig_cpu = tf.identity(w_noncontig)
    w_contig_cpu = tf.identity(w_contig)
    bias_cpu = tf.identity(bias)

    result_cpu_noncontig = tf.linalg.matmul(x_cpu, w_noncontig_cpu, transpose_b=True) + bias_cpu
    result_cpu_contig = tf.linalg.matmul(x_cpu, w_contig_cpu, transpose_b=True) + bias_cpu
    
    match_cpu = tf.reduce_all(tf.abs(result_cpu_noncontig - result_cpu_contig) < 1e-5)
    print(f"CPU contiguous vs non-contiguous match: {match_cpu.numpy()}")
```