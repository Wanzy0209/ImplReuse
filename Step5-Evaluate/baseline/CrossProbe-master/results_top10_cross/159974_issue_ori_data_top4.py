```python
import tensorflow as tf

# Conversion: Function definition remains the same, TensorFlow supports operator overloading
def addcmul_func(x, y, z):
    return x + (y * z)

# Conversion: torch.randn(128).to("xpu") -> tf.random.normal((128,))
# Note: "xpu" is specific to PyTorch for Intel GPUs. 
# In TensorFlow, device placement is handled automatically or via tf.device context.
x = tf.random.normal((128,))
y = tf.random.normal((128,))
z = tf.random.normal((128,))

out = addcmul_func(x, y, z)
print("eager mode passed")

# Conversion: torch.compile -> tf.function
# tf.function creates a callable graph from a Python function for performance optimization.
addcmul_func_compiled = tf.function(addcmul_func)
out = addcmul_func_compiled(x, y, z)
print("tf.function passed")
```