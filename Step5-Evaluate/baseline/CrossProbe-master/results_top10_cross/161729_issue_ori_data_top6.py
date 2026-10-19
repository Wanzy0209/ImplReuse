```python
import tensorflow as tf
import numpy as np

# torch.set_default_device("cuda") is not directly available in TensorFlow.
# We use a device context manager to place operations on the GPU.
with tf.device('/GPU:0'):
    # torch.set_default_device("cuda") (repeated in source)
    batch, in_dim, out_dim = 128, 1024, 4096

    # torch.randn(batch, in_dim, dtype=torch.float)
    # Conversion: tf.random.normal creates a tensor with normal distribution.
    # dtype=torch.float maps to tf.float32.
    x = tf.random.normal([batch, in_dim], dtype=tf.float32)
    w = tf.random.normal([out_dim, in_dim], dtype=tf.float32)

    def linear(x, w):
        # torch.nn.functional.linear(x, w) performs x @ w.T
        # Conversion: tf.linalg.matmul with transpose_b=True
        return tf.linalg.matmul(x, w, transpose_b=True)

    out = linear(x, w)

    # out.shape, out.stride()
    # Conversion: out.shape works directly. Strides are not a property of tf.Tensor,
    # but can be accessed via the numpy interface.
    print(out.shape, out.numpy().strides)

    # torch.einsum("fd,bd->bf", w, x)
    # Conversion: tf.einsum syntax is identical.
    out = tf.einsum("fd,bd->bf", w, x)
    print(out.shape, out.numpy().strides)

    # np.einsum("fd,bd->bf", w.cpu().numpy(), x.cpu().numpy())
    # Conversion: .numpy() copies the tensor data to the host (CPU).
    out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())
    print(out_np.shape, out_np.strides)
```