```python
import tensorflow as tf
import numpy as np

# Conversion: torch.addmm(a, b, c, alpha, beta) computes beta * a + alpha * (b @ c)
# There is no direct single API in TF, so we implement the logic using matmul and arithmetic.
def addmm(a, b, c, alpha=1.0, beta=1.0):
    return beta * a + alpha * tf.linalg.matmul(b, c)

# success
# Conversion: device="mps" -> tf.device('/GPU:0'). 
# Note: TensorFlow uses '/GPU:0' for the primary GPU. On macOS with MPS support, 
# this usually maps to the Metal backend automatically if tensorflow-macos is used.
with tf.device('/GPU:0'):
    # Conversion: torch.rand -> tf.random.uniform (uniform distribution [0, 1))
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 100), dtype=tf.complex64)
    c = tf.random.uniform((100, 300), dtype=tf.complex64)
    out = addmm(a, b, c, alpha=1.0, beta=0.5)

# Conversion: .cpu() -> moving tensors to CPU context
with tf.device('/CPU:0'):
    # We copy the GPU tensors to CPU to perform the CPU calculation
    a_cpu = tf.identity(a)
    b_cpu = tf.identity(b)
    c_cpu = tf.identity(c)
    
    # Calculate result on CPU
    cpu_result = addmm(a_cpu, b_cpu, c_cpu, alpha=1.0, beta=0.5)

# Conversion: torch.testing.assert_close -> np.testing.assert_allclose
# We fetch the numpy values from the tensors to compare
np.testing.assert_allclose(out.numpy(), cpu_result.numpy())

# fails
with tf.device('/GPU:0'):
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 10000), dtype=tf.complex64)
    c = tf.random.uniform((10000, 300), dtype=tf.complex64)
    out = addmm(a, b, c, alpha=1.0, beta=0.5)

with tf.device('/CPU:0'):
    a_cpu = tf.identity(a)
    b_cpu = tf.identity(b)
    c_cpu = tf.identity(c)
    cpu_result = addmm(a_cpu, b_cpu, c_cpu, alpha=1.0, beta=0.5)

np.testing.assert_allclose(out.numpy(), cpu_result.numpy())
```