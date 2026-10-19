```python
import tensorflow as tf
import numpy as np

# PyTorch: torch.set_default_device("cuda")
# TensorFlow: Use a device context manager to place tensors on GPU if available.
# There is no global "set_default_device" that persists outside a context.
device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
with tf.device(device_name):

    # PyTorch: torch.set_default_device("cuda")
    # (Redundant call in source, handled by context above)

    batch, in_dim, out_dim = 128, 1024, 4096

    # PyTorch: x = torch.randn(batch, in_dim, dtype=torch.float)
    # TensorFlow: tf.random.normal creates a tensor with normal distribution.
    x = tf.random.normal((batch, in_dim), dtype=tf.float32)

    # PyTorch: w = torch.randn(out_dim, in_dim, dtype=torch.float)
    w = tf.random.normal((out_dim, in_dim), dtype=tf.float32)

    def linear(x, w):
        # PyTorch: F.linear(x, w) performs x @ w.T
        # TensorFlow: tf.linalg.matmul with transpose_b=True
        return tf.linalg.matmul(x, w, transpose_b=True)

    out = linear(x, w)

    # PyTorch: print(out.shape, out.stride())
    # TensorFlow: tf.Tensor does not have a .stride() method.
    # We convert to numpy to access strides.
    print(out.shape, out.numpy().strides)

    # PyTorch: out = torch.einsum("fd,bd->bf", w, x)
    # TensorFlow: tf.einsum syntax is identical.
    out = tf.einsum("fd,bd->bf", w, x)
    print(out.shape, out.numpy().strides)

    # PyTorch: out_np = np.einsum("fd,bd->bf", w.cpu().numpy(), x.cpu().numpy())
    # TensorFlow: .numpy() converts the tensor to a numpy array.
    out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())
    print(out_np.shape, out_np.strides)
```