```python
import tensorflow as tf

def f(x):
    # PyTorch: x.sin().cos()
    # TensorFlow: math.cos(math.sin(x))
    return tf.math.cos(tf.math.sin(x))


@tf.function
def g(x):
    # PyTorch: torch.vmap(f)(x)
    # TensorFlow: Operations are vectorized by default, so calling f(x) applies the operation element-wise.
    return f(x)


# PyTorch: x = torch.randn(10, device="cuda")
# TensorFlow: tf.random.normal generates a tensor with normal distribution.
# Device placement is handled by TensorFlow runtime (e.g., /GPU:0 if available).
x = tf.random.normal((10,))

g(x)
```