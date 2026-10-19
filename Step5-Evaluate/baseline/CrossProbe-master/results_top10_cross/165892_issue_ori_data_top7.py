```python
import tensorflow as tf

# Conversion: torch.rand -> tf.random.uniform
# Conversion: device="cuda" -> tf.device('/GPU:0')
# Conversion: dtype=torch.float16 -> dtype=tf.float16
with tf.device('/GPU:0'):
    A = tf.random.uniform((1, 1024, 1024), dtype=tf.float16)
    B = tf.random.uniform((1, 1024, 1024), dtype=tf.float16)

# Conversion: @torch.compile -> @tf.function
@tf.function
def linear(weight, input):
    # Conversion: torch.bmm -> tf.linalg.matmul
    # Conversion: out_dtype=torch.float32 -> tf.cast(..., tf.float32)
    # Note: tf.linalg.matmul handles batched matrix multiplication automatically
    return tf.cast(tf.linalg.matmul(input, weight), tf.float32)

linear(A, B)
```