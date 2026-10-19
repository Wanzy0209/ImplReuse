```python
import tensorflow as tf

# Conversion: torch.rand -> tf.random.uniform
# Note: In TensorFlow, device placement is handled implicitly or via tf.device context,
# not as a direct argument in the tensor creation function.
A = tf.random.uniform((1024, 1024), dtype=tf.float16)
B = tf.random.uniform((1024, 1024), dtype=tf.float16)

# Conversion: @torch.compile -> @tf.function
@tf.function
def linear(weight, input):
    # Conversion: torch.mm -> tf.linalg.matmul
    # Conversion: out_dtype=torch.float32 -> tf.cast(..., tf.float32)
    # Note: tf.linalg.matmul output dtype matches input dtype, so we cast the result.
    return tf.cast(tf.linalg.matmul(input, weight), tf.float32)

linear(A, B)
```