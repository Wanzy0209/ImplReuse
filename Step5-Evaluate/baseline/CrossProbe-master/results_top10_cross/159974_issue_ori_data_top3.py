```python
import tensorflow as tf

def addcmul_func(x, y, z):
    return x + (y * z)

# Conversion: torch.randn(128).to("xpu") -> tf.random.normal((128,))
# Note: Explicit device placement to 'xpu' is omitted; TF uses default device or tf.device context.
x = tf.random.normal((128,), dtype=tf.float32)
y = tf.random.normal((128,), dtype=tf.float32)
z = tf.random.normal((128,), dtype=tf.float32)

out = addcmul_func(x, y, z)
print("eager mode passed")

# Conversion: torch.compile -> tf.function
addcmul_func_compiled = tf.function(addcmul_func)
out = addcmul_func_compiled(x, y, z)
print("tf.function passed")
```