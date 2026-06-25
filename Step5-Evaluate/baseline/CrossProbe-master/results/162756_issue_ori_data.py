```python
import tensorflow as tf

# torch._inductor.config.combo_kernels = True
# Conversion: TensorFlow handles kernel fusion and optimization via XLA and the runtime automatically.
# There is no direct user-facing equivalent for this specific inductor config flag.

# @torch.compile
# Conversion: Use tf.function with jit_compile=True to compile the function (XLA).
@tf.function(jit_compile=True)
def fn(x, y, z):
    # x.sum(1) -> tf.reduce_sum(x, axis=1)
    # y.mean(1) -> tf.reduce_mean(y, axis=1)
    # z.cumsum(1) -> tf.cumsum(z, axis=1)
    return tf.reduce_sum(x, axis=1), tf.reduce_mean(y, axis=1), tf.cumsum(z, axis=1)

# device="cuda"
# Conversion: Use tf.device context manager to place tensors on GPU.
# Note: This assumes a GPU is available at '/GPU:0'.
with tf.device('/GPU:0'):
    # torch.rand(...) -> tf.random.uniform(...)
    inps = (
        tf.random.uniform((16, 128)),
        tf.random.uniform((32, 128)),
        tf.random.uniform((32, 256)),
    )

fn(*inps)
```