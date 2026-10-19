```python
import tensorflow as tf

# Conversion: torch.rand -> tf.random.uniform
# Conversion: device="cuda" is handled implicitly by TensorFlow if GPU is available
A = tf.random.uniform((1, 1024, 1024), dtype=tf.float16)
B = tf.random.uniform((1, 1024, 1024), dtype=tf.float16)

# Conversion: @torch.compile -> @tf.function
@tf.function
def linear(weight, input):
    # Conversion: torch.bmm -> tf.matmul
    # Conversion: out_dtype=torch.float32 -> tf.cast(..., tf.float32)
    return tf.cast(tf.matmul(input, weight), tf.float32)

linear(A, B)
```