```python
import tensorflow as tf

# Note: PyTorch's 'mps' device is specific to Apple Silicon. 
# In TensorFlow, device placement is typically implicit or managed via tf.device('/GPU:0').
t = tf.constant([[1.0, 2.0], [2.0, 4.0]])
tf.linalg.lu(t)
```