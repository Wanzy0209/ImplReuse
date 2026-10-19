```python
import tensorflow as tf
import numpy as np

# PyTorch: torch.set_default_device("cuda")
# TensorFlow uses context managers for device placement
with tf.device('/GPU:0'):

    # torch.set_default_device("cuda")
    # Device context is already active above
    batch, in_dim, out_dim = 128, 1024, 4096

    # torch.randn(batch, in_dim, dtype=torch.float)
    # PyTorch default float is float32
    x = tf.random.normal((batch, in_dim), dtype=tf.float32)

    # torch.randn(out_dim, in_dim, dtype=torch.float)
    w = tf.random.normal((out_dim, in_dim), dtype=tf.float32)

    # def linear(x, w): return F.linear(x, w)
    # F.linear in PyTorch computes x * w^T
    def linear(x, w):
        return tf.linalg.matmul(x, w, transpose_b=True)

    out = linear(x, w)

    # print(out.shape, out.stride())
    # TensorFlow tensors do not expose strides in the public API
    print(out.shape, "N/A (strides not exposed)")

    # torch.einsum("fd,bd->bf", w, x)
    out = tf.einsum("fd,bd->bf", w, x)

    # print(out.shape, out.stride())
    print(out.shape, "N/A (strides not exposed)")

    # np.einsum("fd,bd->bf", w.cpu().numpy(), x.cpu().numpy())
    # .numpy() in TensorFlow handles the copy from GPU to CPU automatically
    out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())

    print(out_np.shape, out_np.strides)
```