```python
import tensorflow as tf

def f(real: tf.Tensor, imag: tf.Tensor) -> tf.Tensor:
    z = tf.complex(real, imag)
    # PyTorch's irfft operates on dim=1. TensorFlow's irfft operates on the last axis.
    # We transpose to move dim 1 to the end, apply irfft, then transpose back.
    z_transposed = tf.transpose(z, perm=[0, 2, 1])
    result_transposed = tf.signal.irfft(z_transposed)
    return tf.transpose(result_transposed, perm=[0, 2, 1])

B, F, T = 1, 641, 39

# torch.randn -> tf.random.normal
r_src = tf.random.normal([B, F, T])
i_src = tf.random.normal([B, F, T])

# permute -> transpose
r_mismatch = tf.transpose(r_src, perm=[0, 2, 1])
i_mismatch = tf.transpose(i_src, perm=[0, 2, 1])

# torch.compile -> tf.function
compiled = tf.function(f)

_ = compiled(r_src, i_src)
_ = compiled(r_mismatch, i_mismatch)
```