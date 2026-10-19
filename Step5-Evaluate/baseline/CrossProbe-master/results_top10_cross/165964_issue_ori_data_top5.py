```python
import tensorflow as tf

# Note: TensorFlow uses a context manager for device placement instead of a device argument
with tf.device('/GPU:0'):
    # tf.ones creates a tensor of ones
    # .numpy() converts the tensor to a numpy array, and [0] extracts the scalar value
    # (PyTorch's .item() on a 1-element tensor returns the scalar)
    print(tf.ones(1).numpy()[0])
```