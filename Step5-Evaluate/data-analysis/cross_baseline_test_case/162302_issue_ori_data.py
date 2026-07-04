```python
import tensorflow as tf

# Conversion: torch.tensor -> tf.constant
A = tf.constant([
    [4.0, -3.0,  2.0, -1.0],
    [-1.0,  2.0, -3.0,  4.0],
    [3.0, -4.0,  1.0, -2.0],
    [-2.0,  1.0, -4.0,  3.0]
], dtype=tf.float32)

# Conversion: torch.linalg.tensorinv(A, ind=1) -> tf.linalg.inv(A)
# Note: For a square matrix (4x4) with ind=1, tensorinv is mathematically equivalent to the matrix inverse.
with tf.device('/CPU:0'):
    cpu_result = tf.linalg.inv(A)

# Conversion: A.to("cuda") -> tf.device('/GPU:0')
# Note: Requires a GPU-enabled environment to run without error
with tf.device('/GPU:0'):
    gpu_result = tf.linalg.inv(A)

# Conversion: print(tensor) -> print(tensor.numpy()) to display values
print("CPU:\n", cpu_result.numpy())
print("GPU:\n", gpu_result.numpy())
print("CPU / GPU elementwise ratio:\n", (cpu_result / gpu_result).numpy())
```