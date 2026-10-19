```python
import tensorflow as tf

# Conversion: Function logic is compatible with TensorFlow tensors
def addcmul_func(x, y, z):
    return x + (y * z)

# Conversion: torch.randn(128).to("xpu")
# "xpu" refers to Intel GPU. In TensorFlow, we use standard device placement.
# We place tensors on GPU to mimic the hardware intent.
with tf.device('/GPU:0'):
    x = tf.random.normal((128,))
    y = tf.random.normal((128,))
    z = tf.random.normal((128,))

# Eager execution is default in TensorFlow 2.x
out = addcmul_func(x, y, z)
print("eager mode passed")

# Conversion: torch.compile -> tf.function
# tf.function creates a callable graph from the Python function
addcmul_func_compiled = tf.function(addcmul_func)

out = addcmul_func_compiled(x, y, z)
print("tf.function passed")
```