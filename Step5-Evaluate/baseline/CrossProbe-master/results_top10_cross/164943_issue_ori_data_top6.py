```python
import tensorflow as tf

# Create a tensor
foo = tf.constant([[1]])

# Move tensor to 'mps' (Metal Performance Shaders) device.
# In TensorFlow, device placement is typically handled via context managers.
# On macOS with MPS support, this is usually '/GPU:0'.
with tf.device('/GPU:0'):
    foo = tf.identity(foo)
```