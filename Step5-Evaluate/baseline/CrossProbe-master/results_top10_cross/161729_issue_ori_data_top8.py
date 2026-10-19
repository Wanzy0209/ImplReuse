```python
import tensorflow as tf
import numpy as np

# Conversion: torch.set_default_device("cuda") is handled via tf.device context manager
with tf.device('/GPU:0'):
    # Conversion: torch.set_default_device("cuda") is handled via context
    batch, in_dim, out_dim = 128, 1024, 4096
    
    # Conversion: torch.randn -> tf.random.normal, dtype=torch.float -> tf.float32
    x = tf.random.normal((batch, in_dim), dtype=tf.float32)
    w = tf.random.normal((out_dim, in_dim), dtype=tf.float32)

    def linear(x, w):
        # Conversion: F.linear(x, w) -> tf.linalg.matmul(x, w, transpose_b=True)
        return tf.linalg.matmul(x, w, transpose_b=True)

    out = linear(x, w)
    
    # Conversion: TensorFlow tensors do not have a .stride() method. 
    # Printing shape only.
    print(out.shape) 
    # print(out.stride()) # Not available in TensorFlow
    
    # Conversion: torch.einsum -> tf.einsum
    out = tf.einsum("fd,bd->bf", w, x)
    print(out.shape)
    # print(out.stride()) # Not available in TensorFlow

# Conversion: w.cpu().numpy() -> w.numpy()
# Note: .numpy() copies the tensor to CPU memory if it isn't already there.
out_np = np.einsum("fd,bd->bf", w.numpy(), x.numpy())
print(out_np.shape, out_np.strides)
```