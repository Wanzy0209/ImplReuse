```python
import tensorflow as tf

# PyTorch: device="cuda"
# TensorFlow: Use tf.device context to place tensors on GPU
with tf.device('/GPU:0'):
    # PyTorch: torch.rand((1024, 1024), device="cuda", dtype=torch.float16)
    # TensorFlow: tf.random.uniform generates random values from a uniform distribution
    A = tf.random.uniform((1024, 1024), dtype=tf.float16)
    B = tf.random.uniform((1024, 1024), dtype=tf.float16)

# PyTorch: @torch.compile
# TensorFlow: @tf.function compiles the function into a graph
@tf.function
def linear(weight, input):
    # PyTorch: torch.mm(input, weight, out_dtype=torch.float32)
    # TensorFlow: tf.matmul performs matrix multiplication.
    # Note: tf.matmul does not have an out_dtype argument, so we cast the result.
    return tf.cast(tf.matmul(input, weight), tf.float32)

linear(A, B)
```