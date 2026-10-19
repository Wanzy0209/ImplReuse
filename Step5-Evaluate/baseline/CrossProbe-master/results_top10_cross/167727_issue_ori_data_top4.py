```python
import tensorflow as tf

# success
# Conversion: torch.rand -> tf.random.uniform, device="mps" -> tf.device("/GPU:0")
with tf.device("/GPU:0"):
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 100), dtype=tf.complex64)
    c = tf.random.uniform((100, 300), dtype=tf.complex64)
    # Conversion: torch.addmm(a, b, c, alpha=1.0, beta=0.5) -> beta * a + alpha * tf.matmul(b, c)
    out = 0.5 * a + 1.0 * tf.matmul(b, c)

# Conversion: .cpu() -> tf.identity with CPU device context
with tf.device("/CPU:0"):
    out_gpu_on_cpu = tf.identity(out)

with tf.device("/CPU:0"):
    a_cpu = tf.identity(a)
    b_cpu = tf.identity(b)
    c_cpu = tf.identity(c)
    out_cpu = 0.5 * a_cpu + 1.0 * tf.matmul(b_cpu, c_cpu)

# Conversion: torch.testing.assert_close -> tf.debugging.assert_near
tf.debugging.assert_near(out_gpu_on_cpu, out_cpu)

# fails
with tf.device("/GPU:0"):
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 10000), dtype=tf.complex64)
    c = tf.random.uniform((10000, 300), dtype=tf.complex64)
    out = 0.5 * a + 1.0 * tf.matmul(b, c)

with tf.device("/CPU:0"):
    out_gpu_on_cpu = tf.identity(out)

with tf.device("/CPU:0"):
    a_cpu = tf.identity(a)
    b_cpu = tf.identity(b)
    c_cpu = tf.identity(c)
    out_cpu = 0.5 * a_cpu + 1.0 * tf.matmul(b_cpu, c_cpu)

tf.debugging.assert_near(out_gpu_on_cpu, out_cpu)
```