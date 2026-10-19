```python
import tensorflow as tf

# Conversion: Logic remains the same as operators are overloaded for TF tensors
def addcmul_func(x, y, z):
    return x + (y * z)

# Conversion: torch.randn(128).to("xpu") -> tf.random.normal((128,))
# Note: Explicit device placement (.to("xpu")) is omitted as TensorFlow handles device placement automatically or via context managers.
x = tf.random.normal((128,))
y = tf.random.normal((128,))
z = tf.random.normal((128,))

out = addcmul_func(x, y, z)
print("eager mode passed")

# Conversion: torch.compile -> tf.function
addcmul_func_compiled = tf.function(addcmul_func)
out = addcmul_func_compiled(x, y, z)
print("tf.function passed")
```