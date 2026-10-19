```python
import tensorflow as tf
import numpy as np

# PyTorch: torch.set_default_device("cuda")
# TensorFlow uses context managers to control device placement for operations
with tf.device('/GPU:0'):
    # torch.set_default_device("cuda") # Duplicate line in source
    batch, in_dim, out_dim = 128, 1024, 4096

    # PyTorch: torch.randn(batch, in_dim, dtype=torch.float)
    # TensorFlow: tf.random.normal with dtype tf.float32
    x = tf.random.normal((batch, in_dim), dtype=tf.float32)
    w = tf.random.normal((out_dim, in_dim), dtype=tf.float32)

    def linear(x, w):
        # PyTorch: F.linear(x, w) computes x @ w.T
        # TensorFlow: tf.linalg.matmul with transpose_b=True
        return tf.linalg.matmul(x, w, transpose_b=True)

    out = linear(x, w)
    # PyTorch: out.stride()
    # TensorFlow: Tensors do not have a public stride() method. Printing shape only.
    print(out.shape)
    out = tf.einsum("fd,bd->bf", w, x)
    print(out.shape)

    # PyTorch: w.cpu().numpy()
    # TensorFlow: .numpy() copies the tensor data to the host (CPU) as a numpy array
    out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())
    print(out_np.shape, out_np.strides)
```