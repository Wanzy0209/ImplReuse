```python
import tensorflow as tf

# Conversion: torch.rand with device="cuda" and dtype=torch.float16
# TensorFlow uses tf.random.uniform and explicit device placement
with tf.device('/GPU:0'):
    A = tf.random.uniform((1, 1024, 1024), dtype=tf.float16)
    B = tf.random.uniform((1, 1024, 1024), dtype=tf.float16)

# Conversion: @torch.compile -> @tf.function for graph optimization
@tf.function
def linear(weight, input):
    # Conversion: torch.bmm -> tf.linalg.matmul
    # Conversion: out_dtype=torch.float32 -> explicit cast to tf.float32
    return tf.cast(tf.linalg.matmul(input, weight), tf.float32)

linear(A, B)
```