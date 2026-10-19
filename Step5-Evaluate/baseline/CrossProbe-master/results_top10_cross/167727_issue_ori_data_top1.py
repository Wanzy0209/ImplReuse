```python
import tensorflow as tf

# Helper function to mimic torch.addmm
# Formula: out = beta * input + alpha * (mat1 @ mat2)
def addmm(input_tensor, mat1, mat2, alpha=1.0, beta=1.0):
    return beta * input_tensor + alpha * tf.linalg.matmul(mat1, mat2)

# success
# Conversion: torch.rand -> tf.random.uniform
# Conversion: device="mps" -> tf.device('/GPU:0')
with tf.device('/GPU:0'):
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 100), dtype=tf.complex64)
    c = tf.random.uniform((100, 300), dtype=tf.complex64)
    out = addmm(a, b, c, alpha=1.0, beta=0.5)

# Comparison: Compute on GPU, move to CPU vs Compute on CPU
# Conversion: .cpu() -> tf.identity inside CPU context
# Conversion: torch.testing.assert_close -> tf.debugging.assert_near
with tf.device('/CPU:0'):
    # Move GPU result to CPU
    res_gpu_on_cpu = tf.identity(out)
    # Move inputs to CPU and compute
    a_cpu = tf.identity(a)
    b_cpu = tf.identity(b)
    c_cpu = tf.identity(c)
    res_cpu = addmm(a_cpu, b_cpu, c_cpu, alpha=1.0, beta=0.5)

tf.debugging.assert_near(res_gpu_on_cpu, res_cpu)

# fails
with tf.device('/GPU:0'):
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 10000), dtype=tf.complex64)
    c = tf.random.uniform((10000, 300), dtype=tf.complex64)
    out = addmm(a, b, c, alpha=1.0, beta=0.5)

with tf.device('/CPU:0'):
    res_gpu_on_cpu = tf.identity(out)
    a_cpu = tf.identity(a)
    b_cpu = tf.identity(b)
    c_cpu = tf.identity(c)
    res_cpu = addmm(a_cpu, b_cpu, c_cpu, alpha=1.0, beta=0.5)

tf.debugging.assert_near(res_gpu_on_cpu, res_cpu)
```