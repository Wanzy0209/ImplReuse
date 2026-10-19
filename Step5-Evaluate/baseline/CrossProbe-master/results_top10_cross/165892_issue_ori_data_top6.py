```python
import tensorflow as tf

# Conversion: torch.rand with device and dtype -> tf.random.uniform
# Note: TensorFlow uses GPU automatically if available, or use tf.device('/GPU:0')
A = tf.random.uniform((1, 1024, 1024), dtype=tf.float16)
B = tf.random.uniform((1, 1024, 1024), dtype=tf.float16)

# Conversion: @torch.compile -> @tf.function
@tf.function
def linear(weight, input):
    # Conversion: torch.bmm(input, weight, out_dtype=torch.float32) -> tf.matmul
    # tf.matmul supports batched multiplication.
    # To simulate out_dtype=torch.float32, inputs are cast to float32 before multiplication.
    return tf.matmul(tf.cast(input, tf.float32), tf.cast(weight, tf.float32))

linear(A, B)
```