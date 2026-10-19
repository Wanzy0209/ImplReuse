```python
import tensorflow as tf
import numpy as np

# success
# Conversion: torch.rand -> tf.random.uniform, device="mps" -> tf.device("/GPU:0")
with tf.device("/GPU:0"):
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 100), dtype=tf.complex64)
    c = tf.random.uniform((100, 300), dtype=tf.complex64)
    # Conversion: torch.addmm(a, b, c, alpha=1.0, beta=0.5) -> beta*a + alpha*(b @ c)
    out = 0.5 * a + 1.0 * tf.linalg.matmul(b, c)

# Conversion: torch.testing.assert_close -> np.testing.assert_allclose
# Conversion: .cpu() -> .numpy() (brings tensor to host memory)
# Conversion: Explicit CPU device calculation for the second argument
with tf.device("/CPU:0"):
    # Move tensors to CPU context to simulate a.cpu(), b.cpu(), c.cpu()
    a_cpu = tf.identity(a)
    b_cpu = tf.identity(b)
    c_cpu = tf.identity(c)
    cpu_calc = 0.5 * a_cpu + 1.0 * tf.linalg.matmul(b_cpu, c_cpu)

np.testing.assert_allclose(out.numpy(), cpu_calc.numpy())

# fails
with tf.device("/GPU:0"):
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 10000), dtype=tf.complex64)
    c = tf.random.uniform((10000, 300), dtype=tf.complex64)
    out = 0.5 * a + 1.0 * tf.linalg.matmul(b, c)

with tf.device("/CPU:0"):
    a_cpu = tf.identity(a)
    b_cpu = tf.identity(b)
    c_cpu = tf.identity(c)
    cpu_calc = 0.5 * a_cpu + 1.0 * tf.linalg.matmul(b_cpu, c_cpu)

np.testing.assert_allclose(out.numpy(), cpu_calc.numpy())
```