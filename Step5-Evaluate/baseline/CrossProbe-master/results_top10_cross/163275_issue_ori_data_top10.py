```python
import tensorflow as tf

# Conversion: torch.rand with device="cuda" -> tf.random.uniform inside a device context
# Conversion: torch.float16 -> tf.float16
with tf.device('/GPU:0'):
    A = tf.random.uniform((1024, 1024), dtype=tf.float16)
    B = tf.random.uniform((1024, 1024), dtype=tf.float16)

# Conversion: @torch.compile -> @tf.function
@tf.function
def linear(weight, input):
    # Conversion: torch.mm -> tf.matmul
    # Conversion: out_dtype=torch.float32 -> Cast inputs to tf.float32 to perform computation in higher precision
    return tf.matmul(tf.cast(input, tf.float32), tf.cast(weight, tf.float32))

linear(A, B)
```