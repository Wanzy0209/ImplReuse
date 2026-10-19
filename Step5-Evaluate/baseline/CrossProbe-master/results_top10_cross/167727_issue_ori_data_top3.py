```python
import tensorflow as tf

# Helper function to mimic torch.addmm
# Formula: beta * input + alpha * (mat1 @ mat2)
def addmm(input, mat1, mat2, alpha=1.0, beta=1.0):
    return input * beta + tf.matmul(mat1, mat2) * alpha

# success
# Create tensors on GPU (mimic device="mps")
with tf.device('/GPU:0'):
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 100), dtype=tf.complex64)
    c = tf.random.uniform((100, 300), dtype=tf.complex64)
    # Compute on GPU
    out_gpu = addmm(a, b, c, alpha=1.0, beta=0.5)

# Move tensors to CPU for reference computation
a_cpu = tf.identity(a)
b_cpu = tf.identity(b)
c_cpu = tf.identity(c)
out_cpu = addmm(a_cpu, b_cpu, c_cpu, alpha=1.0, beta=0.5)

# Compare GPU result with CPU result
# tf.debugging.assert_close is the TensorFlow equivalent to torch.testing.assert_close
tf.debugging.assert_close(out_gpu, out_cpu)

# fails
with tf.device('/GPU:0'):
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 10000), dtype=tf.complex64)
    c = tf.random.uniform((10000, 300), dtype=tf.complex64)
    out_gpu = addmm(a, b, c, alpha=1.0, beta=0.5)

a_cpu = tf.identity(a)
b_cpu = tf.identity(b)
c_cpu = tf.identity(c)
out_cpu = addmm(a_cpu, b_cpu, c_cpu, alpha=1.0, beta=0.5)

tf.debugging.assert_close(out_gpu, out_cpu)
```