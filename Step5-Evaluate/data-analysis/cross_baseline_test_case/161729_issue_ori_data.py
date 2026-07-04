```python
import tensorflow as tf
import numpy as np

# torch.set_default_device("cuda")
# TensorFlow uses device placement contexts. We use '/GPU:0' to mimic "cuda".
with tf.device('/GPU:0'):

    # torch.set_default_device("cuda")
    batch, in_dim, out_dim = 128, 1024, 4096
    # torch.randn(batch, in_dim, dtype=torch.float)
    x = tf.random.normal((batch, in_dim), dtype=tf.float32)
    # torch.randn(out_dim, in_dim, dtype=torch.float)
    w = tf.random.normal((out_dim, in_dim), dtype=tf.float32)

    def linear(x, w):
        # torch.nn.functional.linear(x, w) performs x @ w.T
        # Note: The provided documentation for parallel_stack is not applicable for linear operations.
        return tf.linalg.matmul(x, w, transpose_b=True)

    out = linear(x, w)
    # out.stride() is not supported in TensorFlow
    print(out.shape, None)
    out = tf.einsum("fd,bd->bf", w, x)
    print(out.shape, None)
    # w.cpu().numpy() -> w.numpy()
    out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())
    print(out_np.shape, out_np.strides)
```