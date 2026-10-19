```python
import tensorflow as tf
import math

# PyTorch: torch.set_default_device('cuda')
# TensorFlow: Uses device context managers for placement.
# We wrap the tensor creation to ensure it lands on the GPU.
with tf.device('/GPU:0'):
    in_dim = 24
    out_dim = 2
    xavier_stddev = math.sqrt(2.0 / (in_dim + out_dim))
    
    # PyTorch: torch.normal(mean, std, size, dtype)
    # TensorFlow: tf.random.normal(shape, mean, stddev, dtype)
    W = tf.random.normal(shape=(in_dim, out_dim), mean=0.0, stddev=xavier_stddev, dtype=tf.float32)
    
    # PyTorch: W.device
    # TensorFlow: W.device returns the device name string
    print(W.device)
```