```python
import tensorflow as tf

# torch.tensor([[1]]) creates a tensor. In TensorFlow, we use tf.constant for literals.
foo = tf.constant([[1]])

# foo.to('mps') moves the tensor to the MPS device (Apple Silicon GPU).
# TensorFlow does not natively support 'mps' as a device string.
# We map this to the standard GPU device string. In TF, we use a context manager
# to place operations/tensors on a specific device.
with tf.device('/GPU:0'):
    foo = tf.identity(foo)
```