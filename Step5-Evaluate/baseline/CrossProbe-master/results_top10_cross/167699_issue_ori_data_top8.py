```python
import tensorflow as tf

def f(x):
    # PyTorch: x.sin().cos() computes cos(sin(x))
    # TensorFlow: tf.cos(tf.sin(x))
    return tf.cos(tf.sin(x))

# PyTorch: @torch.compile optimizes the function (graph mode)
# TensorFlow: @tf.function converts the function into a graph
@tf.function
def g(x):
    # PyTorch: torch.vmap(f)(x) vectorizes the function f over the input
    # TensorFlow: Ops are natively vectorized, so we can call f(x) directly.
    # If explicit mapping were required for non-vectorized logic, tf.vectorized_map could be used.
    return f(x)

# PyTorch: torch.randn(10, device="cuda") creates random numbers on GPU
# TensorFlow: tf.random.normal creates random numbers, placed on GPU via context manager
with tf.device('/GPU:0'):
    x = tf.random.normal((10,))
    g(x)
```