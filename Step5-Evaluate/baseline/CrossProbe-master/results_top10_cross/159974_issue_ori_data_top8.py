```python
import tensorflow as tf

# Conversion: Define the function logic (x + y * z)
def addcmul_func(x, y, z):
    return x + (y * z)

# Conversion: torch.randn(128).to("xpu") -> tf.random.normal((128,))
# Note: "xpu" is a PyTorch specific device string. In TensorFlow, we use standard device placement 
# (e.g., '/GPU:0' or '/CPU:0'). Here we use the default device.
x = tf.random.normal((128,))
y = tf.random.normal((128,))
z = tf.random.normal((128,))

out = addcmul_func(x, y, z)
print("eager mode passed")

# Conversion: torch.compile -> tf.function
# tf.function compiles a Python function into a callable TensorFlow graph.
addcmul_func_compiled = tf.function(addcmul_func)
out = addcmul_func_compiled(x, y, z)
print("tf.function passed")
```