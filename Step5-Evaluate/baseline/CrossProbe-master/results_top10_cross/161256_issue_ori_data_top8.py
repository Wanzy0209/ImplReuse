```python
import tensorflow as tf

# Conversion: torch.set_default_dtype(torch.float64)
# TensorFlow does not have a global function to set the default dtype.
# We must explicitly specify dtype=tf.float64 in the tensor creation functions below.
dtype = tf.float64

# Conversion: device="cuda"
# We use a device context manager to place tensors on the GPU.
# Note: This requires a CUDA-enabled TensorFlow build and a GPU.
with tf.device('/GPU:0'):
    # Conversion: torch.randn(1000, 1000, device="cuda")
    # tf.random.normal creates a tensor with random values from a normal distribution.
    # Shape is provided as a tuple.
    x = tf.random.normal((1000, 1000), dtype=dtype)
    y = tf.random.normal((1000, 1000), dtype=dtype)

    # Conversion: z = x @ y
    # Matrix multiplication is supported by the @ operator in TensorFlow 2.x.
    z = x @ y

    print(z[0, 0])
```