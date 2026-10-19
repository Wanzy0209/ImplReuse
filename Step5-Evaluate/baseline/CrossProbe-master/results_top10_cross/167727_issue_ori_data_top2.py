```python
import tensorflow as tf
import numpy as np

# Conversion: torch.rand with complex dtype generates complex numbers with uniform [0,1) real/imag parts
def rand_complex(shape, dtype=tf.complex64):
    real = tf.random.uniform(shape, dtype=tf.float32)
    imag = tf.random.uniform(shape, dtype=tf.float32)
    return tf.complex(real, imag)

# Conversion: torch.addmm(input, mat1, mat2, beta=1, alpha=1)
# Formula: output = beta * input + alpha * (mat1 @ mat2)
def addmm(input_tensor, mat1, mat2, alpha=1.0, beta=1.0):
    return beta * input_tensor + alpha * tf.linalg.matmul(mat1, mat2)

# success
# Conversion: device="mps" (Metal) -> device="/GPU:0" (CUDA/ROCm)
with tf.device('/GPU:0'):
    a = rand_complex((64, 300))
    b = rand_complex((64, 100))
    c = rand_complex((100, 300))
    out = addmm(a, b, c, alpha=1.0, beta=0.5)

# Conversion: torch.testing.assert_close -> np.testing.assert_allclose
# Conversion: .cpu() -> .cpu() (moves tensor to CPU memory)
np.testing.assert_allclose(addmm(a, b, c, alpha=1.0, beta=0.5).cpu().numpy(), 
                           addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5).numpy())

# fails
with tf.device('/GPU:0'):
    a = rand_complex((64, 300))
    b = rand_complex((64, 10000))
    c = rand_complex((10000, 300))
    out = addmm(a, b, c, alpha=1.0, beta=0.5)

np.testing.assert_allclose(addmm(a, b, c, alpha=1.0, beta=0.5).cpu().numpy(), 
                           addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5).numpy())
```