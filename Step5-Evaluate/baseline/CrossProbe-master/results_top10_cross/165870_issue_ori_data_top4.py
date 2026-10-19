```python
import tensorflow as tf

# Conversion note: PyTorch 'device="mps"' is specific to Apple Silicon.
# In TensorFlow, device placement is typically handled via context managers (e.g., with tf.device('/GPU:0'):).
# Here we create the tensor on the default device.
t = tf.constant([[1.0, 2.0], [2.0, 4.0]])

# Conversion note: PyTorch's 'linalg.lu_factor' returns a packed LU matrix and pivots.
# TensorFlow's 'linalg.lu' returns the permutation matrix 'p', lower triangular 'l', and upper triangular 'u'.
# The functionality is equivalent but the return format differs.
tf.linalg.lu(t)
```