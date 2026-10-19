```python
import tensorflow as tf

A = tf.constant([
    [4.0, -3.0,  2.0, -1.0],
    [-1.0,  2.0, -3.0,  4.0],
    [3.0, -4.0,  1.0, -2.0],
    [-2.0,  1.0, -4.0,  3.0]
], dtype=tf.float32)

# torch.linalg.tensorinv with ind=1 on a square matrix is equivalent to matrix inverse
with tf.device('/CPU:0'):
    cpu_result = tf.linalg.inv(A)

with tf.device('/GPU:0'):
    gpu_result = tf.linalg.inv(A)

print("CPU:\n", cpu_result)
print("GPU:\n", gpu_result)
# TensorFlow handles cross-device operations (like division) automatically
print("CPU / GPU elementwise ratio:\n", cpu_result / gpu_result)
```