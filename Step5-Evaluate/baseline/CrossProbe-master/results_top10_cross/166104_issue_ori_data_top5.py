```python
import tensorflow as tf
import math

# PyTorch: torch.set_default_device('cuda')
# TensorFlow uses a context manager to specify device placement
with tf.device('/GPU:0'):
    in_dim = 24
    out_dim = 2
    xavier_stddev = math.sqrt(2.0 / (in_dim + out_dim))
    # PyTorch: torch.normal(0, xavier_stddev, size=(in_dim, out_dim), dtype=torch.float32)
    # TensorFlow: tf.random.normal(shape, mean, stddev, dtype)
    W = tf.random.normal(shape=(in_dim, out_dim), mean=0.0, stddev=xavier_stddev, dtype=tf.float32)
    print(W.device)
```