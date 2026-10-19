```python
import tensorflow as tf

def addcmul_func(x, y, z):
    return x + (y * z)

# Conversion: torch.randn -> tf.random.normal
# Conversion: .to("xpu") -> tf.device context. 
# Note: TensorFlow does not natively support "xpu" (Intel GPU) in the standard API.
# We map this to GPU if available, otherwise CPU.
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
with tf.device(device_name):
    x = tf.random.normal((128,))
    y = tf.random.normal((128,))
    z = tf.random.normal((128,))

out = addcmul_func(x, y, z)
print("eager mode passed")

# Conversion: torch.compile -> tf.function
# tf.function creates a callable graph from the Python function, analogous to torch.compile
addcmul_func_compiled = tf.function(addcmul_func)
out = addcmul_func_compiled(x, y, z)
print("tf.function passed")
```