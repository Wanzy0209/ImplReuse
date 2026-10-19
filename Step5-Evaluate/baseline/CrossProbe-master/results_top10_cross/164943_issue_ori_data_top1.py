```python
import tensorflow as tf

# torch.tensor creates a tensor; tf.constant is the TensorFlow equivalent.
foo = tf.constant([[1]])

# PyTorch's .to('mps') moves the tensor to the Apple Silicon GPU.
# In TensorFlow, device placement is handled via context managers.
# On Apple Silicon, the GPU is typically identified as '/device:GPU:0'.
with tf.device('/device:GPU:0'):
    foo = tf.identity(foo)
```