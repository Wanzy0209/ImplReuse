```python
import tensorflow as tf

# PyTorch: torch.tensor([[1]])
# TensorFlow: tf.constant creates a tensor with the specified value.
foo = tf.constant([[1]])

# PyTorch: foo.to('mps')
# TensorFlow: Device placement is handled via context managers.
# 'mps' (Metal Performance Shaders) in PyTorch corresponds to '/GPU:0' in TensorFlow on macOS.
# tf.identity is used to place the existing tensor on the new device.
with tf.device('/GPU:0'):
    foo = tf.identity(foo)
```