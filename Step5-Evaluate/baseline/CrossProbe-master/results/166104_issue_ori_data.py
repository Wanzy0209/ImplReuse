```python
import tensorflow as tf
import math

# TensorFlow does not have a direct global equivalent to torch.set_default_device.
# Device placement is handled automatically (GPU if available) or via tf.device contexts.
# torch.set_default_device('cuda')

in_dim = 24
out_dim = 2
xavier_stddev = math.sqrt(2.0 / (in_dim + out_dim))

# Conversion: torch.normal -> tf.random.normal
W = tf.random.normal(shape=(in_dim, out_dim), mean=0.0, stddev=xavier_stddev, dtype=tf.float32)

print(W.device)
```