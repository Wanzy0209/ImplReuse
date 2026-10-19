```python
import tensorflow as tf

def addcmul_func(x, y, z):
    return x + (y * z)

# Conversion: torch.randn(128).to("xpu") -> tf.random.normal((128,))
# Note: "xpu" is PyTorch specific for Intel GPUs. In TensorFlow, tensors are created 
# on the available device (GPU/CPU) by default.
x = tf.random.normal((128,))
y = tf.random.normal((128,))
z = tf.random.normal((128,))

out = addcmul_func(x, y, z)
print("eager mode passed")

# Conversion: torch.compile -> tf.function
# tf.function compiles the Python function into a static TensorFlow graph for performance.
addcmul_func_compiled = tf.function(addcmul_func)
out = addcmul_func_compiled(x, y, z)
print("tf.function passed")
```