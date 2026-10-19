```python
import tensorflow as tf

foo = tf.constant([[1]])  # torch.tensor([[1]])

# TensorFlow uses device contexts for placement; 'mps' maps to '/GPU:0' on macOS
with tf.device('/GPU:0'):
    foo = tf.identity(foo)  # foo = foo.to('mps')
```