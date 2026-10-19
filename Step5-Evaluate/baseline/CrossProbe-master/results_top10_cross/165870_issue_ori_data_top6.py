```python
import tensorflow as tf

# Note: 'mps' device is PyTorch specific. In TensorFlow, device placement is handled
# via context managers (e.g., with tf.device('/GPU:0'):). Defaulting to default device.
t = tf.constant([[1.0, 2.0], [2.0, 4.0]])

# PyTorch's lu_factor returns (LU, pivots).
# TensorFlow's lu returns (lu, p) where p is the permutation matrix.
tf.linalg.lu(t)
```