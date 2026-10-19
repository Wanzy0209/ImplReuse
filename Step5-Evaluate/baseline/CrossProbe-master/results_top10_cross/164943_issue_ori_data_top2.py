```python
import tensorflow as tf

foo = tf.constant([[1]])

# Conversion: PyTorch uses .to('mps') to move tensors to the Metal (GPU) device.
# TensorFlow uses device context managers. 'mps' maps to '/device:GPU:0' on macOS.
with tf.device('/device:GPU:0'):
    foo = tf.identity(foo)
```