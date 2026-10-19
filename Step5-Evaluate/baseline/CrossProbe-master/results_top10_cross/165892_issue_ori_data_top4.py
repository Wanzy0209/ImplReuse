```python
import tensorflow as tf

# PyTorch: torch.rand((1, 1024, 1024), device="cuda", dtype=torch.float16)
# TensorFlow uses tf.random.uniform for random initialization and tf.device for placement
with tf.device('/GPU:0'):
    A = tf.random.uniform((1, 1024, 1024), dtype=tf.float16)
    B = tf.random.uniform((1, 1024, 1024), dtype=tf.float16)

# PyTorch: @torch.compile
# TensorFlow uses @tf.function to compile the function into a graph
@tf.function
def linear(weight, input):
    # PyTorch: torch.bmm(input, weight, out_dtype=torch.float32)
    # TensorFlow uses tf.linalg.matmul for batched matrix multiplication.
    # The output is explicitly cast to float32 to match out_dtype.
    return tf.cast(tf.linalg.matmul(input, weight), tf.float32)

linear(A, B)
```