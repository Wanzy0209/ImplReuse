```python
import tensorflow as tf

# torch._inductor.config.combo_kernels = True
# Conversion: TensorFlow uses XLA for kernel fusion and optimization, 
# enabled via jit_compile=True in tf.function.

@tf.function(jit_compile=True) # Conversion: Equivalent to @torch.compile
def fn(x, y, z):
    # Conversion: x.sum(1) -> tf.reduce_sum(x, axis=1)
    # Conversion: y.mean(1) -> tf.reduce_mean(y, axis=1)
    # Conversion: z.cumsum(1) -> tf.math.cumsum(z, axis=1)
    return tf.reduce_sum(x, axis=1), tf.reduce_mean(y, axis=1), tf.math.cumsum(z, axis=1)

# Conversion: torch.rand(..., device="cuda") -> tf.random.uniform(...) on GPU
with tf.device('/GPU:0'):
    inps = (
        tf.random.uniform((16, 128)),
        tf.random.uniform((32, 128)),
        tf.random.uniform((32, 256)),
    )

fn(*inps)
```