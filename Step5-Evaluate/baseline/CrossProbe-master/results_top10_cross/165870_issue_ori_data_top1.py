```python
import tensorflow as tf

# Conversion: torch.tensor -> tf.constant
# Note: device="mps" is PyTorch specific for Apple Silicon. 
# In TensorFlow, device placement is handled automatically or via tf.device context.
t = tf.constant([[1.0, 2.0], [2.0, 4.0]])

# Conversion: torch.linalg.lu_factor -> tf.linalg.lu
# Note: PyTorch returns (LU, pivots), while TensorFlow returns (lu, p) where p is the permutation matrix.
tf.linalg.lu(t)
```