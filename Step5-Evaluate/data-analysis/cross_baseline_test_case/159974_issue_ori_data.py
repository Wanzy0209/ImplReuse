```python
import tensorflow as tf

def addcmul_func(x, y, z):
    return x + (y * z)

# Conversion: torch.randn(128).to("xpu") -> tf.random.normal([128])
# Note: Explicit device placement (.to("xpu")) is handled implicitly by TensorFlow or via tf.device context.
x = tf.random.normal([128])
y = tf.random.normal([128])
z = tf.random.normal([128])

out = addcmul_func(x, y, z)
print("eager mode passed")

# Conversion: torch.compile -> tf.function
# tf.function creates a callable graph from the Python function, analogous to torch.compile.
addcmul_func_compiled = tf.function(addcmul_func)
out = addcmul_func_compiled(x, y, z)
print("torch.compile passed")
```