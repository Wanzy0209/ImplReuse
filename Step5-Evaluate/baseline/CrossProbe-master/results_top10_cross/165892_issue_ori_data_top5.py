```python
import tensorflow as tf

# Conversion: torch.rand with device="cuda" and dtype=torch.float16
# TensorFlow uses GPU by default if available, explicit placement ensures consistency
with tf.device('/GPU:0'):
    A = tf.random.uniform((1, 1024, 1024), dtype=tf.float16)
    B = tf.random.uniform((1, 1024, 1024), dtype=tf.float16)

# Conversion: @torch.compile -> @tf.function
@tf.function
def linear(weight, input):
    # Conversion: torch.bmm(input, weight, out_dtype=torch.float32)
    # tf.linalg.matmul supports batched matrix multiplication for 3D tensors
    # Explicit cast to tf.float32 matches the out_dtype behavior
    return tf.cast(tf.linalg.matmul(input, weight), tf.float32)

linear(A, B)
```