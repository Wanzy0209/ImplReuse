```python
import tensorflow as tf

def addcmul_func(x, y, z):
    return x + (y * z)

# Conversion: torch.randn -> tf.random.normal
# Conversion: .to("xpu") -> tf.device context manager
# Note: "xpu" is specific to Intel extensions in PyTorch. 
# We use '/GPU:0' here as the standard equivalent for device placement.
with tf.device('/GPU:0'):
    x = tf.random.normal((128,))
    y = tf.random.normal((128,))
    z = tf.random.normal((128,))

out = addcmul_func(x, y, z)
print("eager mode passed")

# Conversion: torch.compile -> tf.function
# tf.function creates a callable graph from a Python function
addcmul_func_compiled = tf.function(addcmul_func)
out = addcmul_func_compiled(x, y, z)
print("tf.function passed")
```