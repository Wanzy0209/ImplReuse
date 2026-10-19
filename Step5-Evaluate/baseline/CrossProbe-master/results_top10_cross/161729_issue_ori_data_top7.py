```python
import tensorflow as tf
import numpy as np

# PyTorch: torch.set_default_device("cuda")
# TensorFlow uses device context managers to specify where operations run
with tf.device('/GPU:0'):
    # torch.set_default_device("cuda") # Redundant line in source, handled by context

    batch, in_dim, out_dim = 128, 1024, 4096
    # torch.randn -> tf.random.normal
    # torch.float -> tf.float32
    x = tf.random.normal((batch, in_dim), dtype=tf.float32)
    w = tf.random.normal((out_dim, in_dim), dtype=tf.float32)

    def linear(x, w):
        # PyTorch F.linear(x, w) computes x @ w.T
        # TensorFlow equivalent is tf.linalg.matmul with transpose_b=True
        return tf.linalg.matmul(x, w, transpose_b=True)

    out = linear(x, w)
    # PyTorch: out.stride() is a method. TensorFlow: out.strides is a property.
    print(out.shape, out.strides)
    
    # torch.einsum -> tf.einsum
    out = tf.einsum("fd,bd->bf", w, x)
    print(out.shape, out.strides)
    
    # PyTorch: .cpu().numpy() moves data to CPU then converts to numpy
    # TensorFlow: .numpy() automatically handles the copy to CPU if necessary
    out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())
    print(out_np.shape, out_np.strides)
```