```python
import tensorflow as tf
import numpy as np

# Conversion: torch.set_default_device is handled via context manager in TF
with tf.device('/GPU:0'):
    batch, in_dim, out_dim = 128, 1024, 4096
    # Conversion: torch.randn -> tf.random.normal
    x = tf.random.normal((batch, in_dim), dtype=tf.float32)
    w = tf.random.normal((out_dim, in_dim), dtype=tf.float32)

    def linear(x, w):
        # Conversion: F.linear(x, w) -> tf.linalg.matmul(x, w, transpose_b=True)
        return tf.linalg.matmul(x, w, transpose_b=True)

    out = linear(x, w)
    # Conversion: TF tensors do not have a .stride() attribute
    print(out.shape, None)
    # Conversion: torch.einsum -> tf.einsum
    out = tf.einsum("fd,bd->bf", w, x)
    print(out.shape, None)
    # Conversion: .cpu().numpy() -> .numpy() (handles device transfer automatically)
    out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())
    print(out_np.shape, out_np.strides)
```