```python
import tensorflow as tf

# Note: PyTorch 'mps' device is specific to Apple Silicon. 
# In TensorFlow, device placement is handled by the runtime or explicit context (e.g., with tf.device('/GPU:0')).
t = tf.constant([[1.0, 2.0], [2.0, 4.0]])

# Note: torch.linalg.lu_factor returns (LU, pivots).
# tf.linalg.lu returns (LU, p) where p is the permutation matrix.
tf.linalg.lu(t)
```