```python
import tensorflow as tf
import numpy as np

# Helper function to generate complex random numbers similar to torch.rand with complex dtype
def rand_complex(shape):
    return tf.complex(tf.random.uniform(shape), tf.random.uniform(shape))

# success
# Conversion: device="mps" (PyTorch) -> tf.device('/GPU:0') (TensorFlow)
with tf.device('/GPU:0'):
    a = rand_complex((64, 300))
    b = rand_complex((64, 100))
    c = rand_complex((100, 300))
    # Conversion: torch.addmm(a, b, c, alpha=1.0, beta=0.5) -> beta*a + alpha*matmul(b, c)
    out = 0.5 * a + 1.0 * tf.linalg.matmul(b, c)

# Conversion: torch.testing.assert_close -> np.testing.assert_allclose
# Conversion: .cpu() -> Explicit computation on CPU device
res_gpu = 0.5 * a + 1.0 * tf.linalg.matmul(b, c)
with tf.device('/CPU:0'):
    res_cpu = 0.5 * a + 1.0 * tf.linalg.matmul(b, c)

np.testing.assert_allclose(res_gpu.numpy(), res_cpu.numpy())

# fails
with tf.device('/GPU:0'):
    a = rand_complex((64, 300))
    b = rand_complex((64, 10000))
    c = rand_complex((10000, 300))
    out = 0.5 * a + 1.0 * tf.linalg.matmul(b, c)

res_gpu = 0.5 * a + 1.0 * tf.linalg.matmul(b, c)
with tf.device('/CPU:0'):
    res_cpu = 0.5 * a + 1.0 * tf.linalg.matmul(b, c)

np.testing.assert_allclose(res_gpu.numpy(), res_cpu.numpy())
```