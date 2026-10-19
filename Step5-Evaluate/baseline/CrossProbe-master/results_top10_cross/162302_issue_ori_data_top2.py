```python
import tensorflow as tf

# torch.tensor -> tf.constant
A = tf.constant([
    [4.0, -3.0,  2.0, -1.0],
    [-1.0,  2.0, -3.0,  4.0],
    [3.0, -4.0,  1.0, -2.0],
    [-2.0,  1.0, -4.0,  3.0]
], dtype=tf.float32)

# torch.linalg.tensorinv(A, ind=1)
# For a 2D square matrix with ind=1, tensorinv is equivalent to the matrix inverse.
# TensorFlow uses tf.linalg.inv for matrix inversion.
cpu_result = tf.linalg.inv(A)

# torch.linalg.tensorinv(A.to("cuda"), ind=1)
# In TensorFlow, we use a device context manager to place operations on the GPU.
with tf.device('/GPU:0'):
    # Explicitly copy A to the GPU context to mimic A.to("cuda")
    A_gpu = tf.identity(A)
    gpu_result = tf.linalg.inv(A_gpu)

print("CPU:\n", cpu_result)
print("GPU:\n", gpu_result)
# Division works across devices in TF eager mode, moving GPU result to CPU implicitly or explicitly
print("CPU / GPU elementwise ratio:\n", cpu_result / gpu_result)
```