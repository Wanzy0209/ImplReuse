```python
import tensorflow as tf

# Helper function to mimic torch.rand for complex numbers
# torch.rand with complex dtype generates real and imaginary parts in [0, 1)
def rand_complex(shape):
    real = tf.random.uniform(shape, dtype=tf.float32)
    imag = tf.random.uniform(shape, dtype=tf.float32)
    return tf.complex(real, imag)

# Helper function to mimic torch.addmm
# Formula: output = beta * input + alpha * (mat1 @ mat2)
def addmm(input_tensor, mat1, mat2, alpha=1.0, beta=1.0):
    return beta * input_tensor + alpha * tf.linalg.matmul(mat1, mat2)

# success
# Using /GPU:0 to simulate the "mps" device
with tf.device('/GPU:0'):
    a = rand_complex((64, 300))
    b = rand_complex((64, 100))
    c = rand_complex((100, 300))
    out = addmm(a, b, c, alpha=1.0, beta=0.5)

# Move inputs to CPU and compute there for comparison
with tf.device('/CPU:0'):
    a_cpu = tf.identity(a)
    b_cpu = tf.identity(b)
    c_cpu = tf.identity(c)
    out_cpu = addmm(a_cpu, b_cpu, c_cpu, alpha=1.0, beta=0.5)

# Compare GPU result (implicitly moved to CPU) vs CPU result
tf.debugging.assert_near(out, out_cpu)

# fails
with tf.device('/GPU:0'):
    a = rand_complex((64, 300))
    b = rand_complex((64, 10000))
    c = rand_complex((10000, 300))
    out = addmm(a, b, c, alpha=1.0, beta=0.5)

with tf.device('/CPU:0'):
    a_cpu = tf.identity(a)
    b_cpu = tf.identity(b)
    c_cpu = tf.identity(c)
    out_cpu = addmm(a_cpu, b_cpu, c_cpu, alpha=1.0, beta=0.5)

tf.debugging.assert_near(out, out_cpu)
```