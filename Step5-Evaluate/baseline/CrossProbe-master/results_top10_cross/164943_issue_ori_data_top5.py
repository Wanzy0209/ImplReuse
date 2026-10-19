```python
import tensorflow as tf

foo = tf.constant([[1]])

# Conversion Note: PyTorch's .to('mps') moves the tensor to the Apple Silicon GPU.
# In TensorFlow, device placement is handled via context managers.
# For Apple Silicon, the device is typically '/GPU:0'.
# Since TF tensors are immutable, we use tf.identity to place the tensor on the device.
with tf.device('/GPU:0'):
    foo = tf.identity(foo)
```