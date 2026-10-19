```python
import tensorflow as tf
import numpy as np

# PyTorch: torch.set_default_device("cuda")
# TensorFlow: Use device context manager to place tensors on GPU
with tf.device('/GPU:0'):
    batch, in_dim, out_dim = 128, 1024, 4096

    # PyTorch: torch.randn(...) -> TensorFlow: tf.random.normal(...)
    # PyTorch: dtype=torch.float -> TensorFlow: dtype=tf.float32
    x = tf.random.normal((batch, in_dim), dtype=tf.float32)
    w = tf.random.normal((out_dim, in_dim), dtype=tf.float32)

    def linear(x, w):
        # PyTorch: F.linear(x, w) performs x @ w.T
        # TensorFlow: tf.linalg.matmul with transpose_b=True
        return tf.linalg.matmul(x, w, transpose_b=True)

    out = linear(x, w)
    print(out.shape, out.strides)

    # PyTorch: torch.einsum -> TensorFlow: tf.einsum
    out = tf.einsum("fd,bd->bf", w, x)
    print(out.shape, out.strides)

    # PyTorch: .cpu().numpy() -> TensorFlow: .numpy() (handles GPU to CPU transfer automatically)
    out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())
    print(out_np.shape, out_np.strides)
```