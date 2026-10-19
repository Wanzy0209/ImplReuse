```python
import tensorflow as tf

# Conversion: torch.rand with complex64
# Generates random complex numbers where real and imaginary parts are uniform [0, 1)
def rand_complex(shape):
    return tf.complex(tf.random.uniform(shape), tf.random.uniform(shape))

# Conversion: torch.addmm(input, mat1, mat2, alpha, beta)
# Performs: beta * input + alpha * (mat1 @ mat2)
def addmm(input, mat1, mat2, alpha=1.0, beta=1.0):
    return beta * input + alpha * tf.linalg.matmul(mat1, mat2)

# Conversion: torch.testing.assert_close
# Checks if two tensors are close element-wise
def assert_close(actual, expected):
    # Default tolerances similar to torch.testing.assert_close
    return tf.debugging.assert_near(actual, expected, rtol=1e-05, atol=1e-08)

# success
# Conversion: device="mps" -> '/GPU:0'
with tf.device('/GPU:0'):
    a = rand_complex((64, 300))
    b = rand_complex((64, 100))
    c = rand_complex((100, 300))
    out = addmm(a, b, c, alpha=1.0, beta=0.5)

# Compare GPU result (moved to CPU) with CPU result
# tf.identity inside the CPU device block forces data transfer
with tf.device('/CPU:0'):
    assert_close(tf.identity(out), addmm(tf.identity(a), tf.identity(b), tf.identity(c), alpha=1.0, beta=0.5))

# fails
with tf.device('/GPU:0'):
    a = rand_complex((64, 300))
    b = rand_complex((64, 10000))
    c = rand_complex((10000, 300))
    out = addmm(a, b, c, alpha=1.0, beta=0.5)

with tf.device('/CPU:0'):
    assert_close(tf.identity(out), addmm(tf.identity(a), tf.identity(b), tf.identity(c), alpha=1.0, beta=0.5))
```