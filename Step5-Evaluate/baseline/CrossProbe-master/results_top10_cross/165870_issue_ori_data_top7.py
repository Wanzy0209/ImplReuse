```python
import tensorflow as tf

# PyTorch 'mps' device is specific to Apple Silicon. 
# In TensorFlow, device placement is typically handled via context managers or automatically.
t = tf.constant([[1.0, 2.0], [2.0, 4.0]])

# tf.linalg.lu computes the LU decomposition.
# Note: PyTorch returns pivot indices, while TensorFlow returns the permutation matrix.
tf.linalg.lu(t)
```