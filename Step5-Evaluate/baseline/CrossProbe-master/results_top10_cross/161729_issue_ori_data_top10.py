```python
import tensorflow as tf
import numpy as np

# torch.set_default_device("cuda")
# TensorFlow does not have a global set_default_device.
# We use a device context manager to place operations on GPU if available.
with tf.device('/GPU:0'):

    # torch.set_default_device("cuda")
    batch, in_dim, out_dim = 128, 1024, 4096

    # x = torch.randn(batch, in_dim, dtype=torch.float)
    x = tf.random.normal((batch, in_dim), dtype=tf.float32)

    # w = torch.randn(out_dim, in_dim, dtype=torch.float)
    w = tf.random.normal((out_dim, in_dim), dtype=tf.float32)

    def linear(x, w):
        # PyTorch F.linear(x, w) performs x @ w.T
        return tf.linalg.matmul(x, w, transpose_b=True)

    out = linear(x, w)

    # PyTorch: print(out.shape, out.stride())
    # TensorFlow tensors do not expose memory strides directly.
    print(out.shape)

    # PyTorch: out = torch.einsum("fd,bd->bf", w, x)
    out = tf.einsum("fd,bd->bf", w, x)

    # PyTorch: print(out.shape, out.stride())
    print(out.shape)

    # PyTorch: out_np = np.einsum("fd,bd->bf", w.cpu().numpy(), x.cpu().numpy())
    # TensorFlow: .numpy() copies the tensor data to the host (CPU) as a numpy array.
    out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())
    print(out_np.shape, out_np.strides)
```