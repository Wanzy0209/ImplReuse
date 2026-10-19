```python
import tensorflow as tf
import numpy as np

# Conversion: torch.set_default_device("cuda")
# TensorFlow uses a context manager for device placement rather than a global default setting.
# We wrap the logic to ensure execution on GPU if available.
with tf.device('/GPU:0'):
    # torch.set_default_device("cuda") # Redundant in source, omitted here
    
    batch, in_dim, out_dim = 128, 1024, 4096
    
    # Conversion: torch.randn -> tf.random.normal
    x = tf.random.normal((batch, in_dim), dtype=tf.float32)
    w = tf.random.normal((out_dim, in_dim), dtype=tf.float32)

    def linear(x, w):
        # Conversion: F.linear(x, w) performs matrix multiplication x @ w.T
        # TensorFlow equivalent is tf.linalg.matmul with transpose_b=True
        return tf.linalg.matmul(x, w, transpose_b=True)

    out = linear(x, w)
    
    # Conversion: out.stride() (method) -> out.strides (property)
    print(out.shape, out.strides)
    
    # Conversion: torch.einsum -> tf.einsum
    out = tf.einsum("fd,bd->bf", w, x)
    print(out.shape, out.strides)
    
    # Conversion: .cpu().numpy() -> .numpy()
    # In TensorFlow eager execution, .numpy() handles the device-to-host copy automatically.
    out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())
    print(out_np.shape, out_np.strides)
```