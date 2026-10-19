```python
import tensorflow as tf

# Create a constant tensor
foo = tf.constant([[1]])

# Move tensor to device. PyTorch 'mps' (Metal) maps to '/device:GPU:0' in TensorFlow.
# tf.identity is used to create a copy of the tensor on the specified device.
with tf.device('/device:GPU:0'):
    foo = tf.identity(foo)
```