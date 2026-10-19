```python
import tensorflow as tf

# Note: 'mps' device is PyTorch specific for Apple Silicon. 
# In TensorFlow, device placement is usually managed via 'tf.device' contexts.
t = tf.constant([[1.0, 2.0], [2.0, 4.0]])

# tf.linalg.lu returns (lu, p) which corresponds to torch.linalg.lu_factor output
tf.linalg.lu(t)
```