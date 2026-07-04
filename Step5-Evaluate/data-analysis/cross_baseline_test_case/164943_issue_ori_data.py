```python
import tensorflow as tf

# torch.tensor creates a tensor; tf.constant is the equivalent for fixed values
foo = tf.constant([[1]])

# PyTorch's .to('mps') moves the tensor to the Apple Silicon GPU.
# In TensorFlow, device placement is handled via context managers.
# Note: TensorFlow typically maps MPS support to '/GPU:0'.
with tf.device('/GPU:0'):
    foo = tf.identity(foo)
```