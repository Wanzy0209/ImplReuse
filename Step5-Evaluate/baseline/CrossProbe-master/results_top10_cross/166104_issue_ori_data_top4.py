```python
import tensorflow as tf
import math

# PyTorch sets a global default device. In TensorFlow, device placement is typically 
# handled via context managers or automatic placement. We use a context manager 
# here to ensure the tensor is created on the GPU, mimicking the source behavior.
with tf.device('/GPU:0'):
    in_dim = 24
    out_dim = 2        
    xavier_stddev = math.sqrt(2.0 / (in_dim + out_dim))
    
    # Conversion: torch.normal(mean, std, size) -> tf.random.normal(shape, mean, stddev)
    W = tf.random.normal(shape=[in_dim, out_dim], mean=0.0, stddev=xavier_stddev, dtype=tf.float32)
    print(W.device)
```