```python
import tensorflow as tf

# Create tensor
foo = tf.constant([[1]])

# Move to device. PyTorch's 'mps' is specific to Apple Silicon.
# In TensorFlow, we use a context manager to place operations on a device (e.g., '/GPU:0').
# Since TF tensors are immutable, we re-create the tensor on the target device.
with tf.device('/GPU:0'):
    foo = tf.constant([[1]])
```