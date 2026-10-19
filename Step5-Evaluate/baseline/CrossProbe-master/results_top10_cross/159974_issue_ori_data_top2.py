```python
import tensorflow as tf

def addcmul_func(x, y, z):
    return x + (y * z)

# Conversion: torch.randn(128) -> tf.random.normal([128])
# Note: .to("xpu") is specific to PyTorch Intel extensions. 
# In TensorFlow, tensors are created on the default available device.
x = tf.random.normal([128])
y = tf.random.normal([128])
z = tf.random.normal([128])

out = addcmul_func(x, y, z)
print("eager mode passed")

# Conversion: torch.compile -> tf.function
# tf.function compiles the function into a static graph for performance.
addcmul_func_compiled = tf.function(addcmul_func)
out = addcmul_func_compiled(x, y, z)
print("torch.compile passed")
```