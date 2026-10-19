```python
import tensorflow as tf
import math

# PyTorch sets default device globally; TensorFlow uses context managers for device placement.
with tf.device('/GPU:0'):
    in_dim = 24
    out_dim = 2
    xavier_stddev = math.sqrt(2.0 / (in_dim + out_dim))
    # torch.normal -> tf.random.normal
    W = tf.random.normal(shape=(in_dim, out_dim), mean=0.0, stddev=xavier_stddev, dtype=tf.float32)
    print(W.device)
```