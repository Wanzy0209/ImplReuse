```python
import tensorflow as tf

# PyTorch: foo = torch.tensor([[1]])
foo = tf.constant([[1]])

# PyTorch: foo = foo.to('mps')
# TensorFlow: Device placement is handled via context managers.
# 'mps' (Metal) maps to '/GPU:0' in TensorFlow.
with tf.device('/GPU:0'):
    foo = tf.identity(foo)
```