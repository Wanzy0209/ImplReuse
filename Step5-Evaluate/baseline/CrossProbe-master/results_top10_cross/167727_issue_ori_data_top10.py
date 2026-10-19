```python
import tensorflow as tf
import numpy as np

# success
# Conversion: device="mps" mapped to '/GPU:0' for TensorFlow
with tf.device('/GPU:0'):
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 100), dtype=tf.complex64)
    c = tf.random.uniform((100, 300), dtype=tf.complex64)
    # Conversion: torch.addmm(a, b, c, alpha=1.0, beta=0.5) -> beta*a + alpha*(b@c)
    out = 0.5 * a + 1.0 * tf.linalg.matmul(b, c)

# Compare GPU computation result with CPU computation result
# torch.addmm(...).cpu() computes on GPU then moves result to CPU
# torch.addmm(a.cpu(), ...) moves inputs to CPU then computes
np.testing.assert_allclose((0.5 * a + 1.0 * tf.linalg.matmul(b, c)).numpy(), 
                           (0.5 * tf.identity(a) + 1.0 * tf.linalg.matmul(tf.identity(b), tf.identity(c))).numpy())

# fails
with tf.device('/GPU:0'):
    a = tf.random.uniform((64, 300), dtype=tf.complex64)
    b = tf.random.uniform((64, 10000), dtype=tf.complex64)
    c = tf.random.uniform((10000, 300), dtype=tf.complex64)
    out = 0.5 * a + 1.0 * tf.linalg.matmul(b, c)

np.testing.assert_allclose((0.5 * a + 1.0 * tf.linalg.matmul(b, c)).numpy(), 
                           (0.5 * tf.identity(a) + 1.0 * tf.linalg.matmul(tf.identity(b), tf.identity(c))).numpy())
```