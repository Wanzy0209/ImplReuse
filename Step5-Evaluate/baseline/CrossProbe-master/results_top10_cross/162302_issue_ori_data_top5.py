```python
import tensorflow as tf

# Define the matrix
A = tf.constant([
    [4.0, -3.0,  2.0, -1.0],
    [-1.0,  2.0, -3.0,  4.0],
    [3.0, -4.0,  1.0, -2.0],
    [-2.0,  1.0, -4.0,  3.0]
], dtype=tf.float32)

# Calculate inverse on CPU
# Conversion: torch.linalg.tensorinv(A, ind=1) for a square matrix is equivalent to tf.linalg.inv(A)
with tf.device('/CPU:0'):
    cpu_result = tf.linalg.inv(A)

# Calculate inverse on GPU
# Conversion: PyTorch moves A to GPU. In TF, we place the operation on GPU, which handles data transfer.
with tf.device('/GPU:0'):
    gpu_result = tf.linalg.inv(A)

# Print results
# Conversion: Use .numpy() to retrieve values for printing, similar to PyTorch's default print behavior for tensors
print("CPU:\n", cpu_result.numpy())
print("GPU:\n", gpu_result.numpy())
print("CPU / GPU elementwise ratio:\n", (cpu_result / gpu_result).numpy())
```