```python
import tensorflow as tf

# PyTorch's device="mps" (Metal Performance Shaders) is specific to Apple Silicon.
# In TensorFlow, device placement is implicit or managed via tf.device contexts.
t = tf.constant([[1.0, 2.0], [2.0, 4.0]])

# PyTorch's lu_factor returns a packed LU matrix and pivots.
# TensorFlow's linalg.lu returns the permutation matrix P, lower triangular L, and upper triangular U.
tf.linalg.lu(t)
```