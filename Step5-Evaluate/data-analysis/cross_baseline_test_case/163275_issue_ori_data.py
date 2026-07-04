```python
import tensorflow as tf

# PyTorch: device="cuda" -> TensorFlow: tf.device('/GPU:0')
# Note: This assumes a GPU is available at index 0.
with tf.device('/GPU:0'):
    # PyTorch: torch.rand((1024, 1024), dtype=torch.float16)
    # TensorFlow: tf.random.uniform((1024, 1024), dtype=tf.float16)
    A = tf.random.uniform((1024, 1024), dtype=tf.float16)
    B = tf.random.uniform((1024, 1024), dtype=tf.float16)

    # PyTorch: @torch.compile -> TensorFlow: @tf.function
    @tf.function
    def linear(weight, input):
        # PyTorch: torch.mm(input, weight, out_dtype=torch.float32)
        # TensorFlow: tf.linalg.matmul(input, weight, output_type=tf.float32)
        return tf.linalg.matmul(input, weight, output_type=tf.float32)

    linear(A, B)
```