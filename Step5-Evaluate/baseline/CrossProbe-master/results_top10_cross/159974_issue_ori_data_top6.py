```python
import tensorflow as tf

def addcmul_func(x, y, z):
    return x + (y * z)

# Conversion: torch.randn(128) -> tf.random.normal((128,))
# Conversion: .to("xpu") is omitted as TF handles device placement automatically or via tf.device context
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