```python
import tensorflow as tf

# Note: PyTorch's "mps" device is specific to Apple Silicon.
# TensorFlow handles device placement differently (e.g., via context managers or runtime defaults).
# We create the tensor on the default device here.
t = tf.constant([[1.0, 2.0], [2.0, 4.0]])

# tf.linalg.lu returns a tuple (lu, p) representing the LU decomposition,
# analogous to torch.linalg.lu_factor.
tf.linalg.lu(t)
```