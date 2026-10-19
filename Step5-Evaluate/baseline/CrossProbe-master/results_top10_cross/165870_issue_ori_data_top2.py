```python
import tensorflow as tf

# Note: PyTorch's device="mps" (Metal Performance Shaders) is specific to PyTorch.
# In TensorFlow, device placement is typically handled via `with tf.device('/GPU:0')` context
# or relies on the default device configuration.
t = tf.constant([[1.0, 2.0], [2.0, 4.0]])

# PyTorch's torch.linalg.lu_factor computes the LU decomposition.
# TensorFlow's equivalent is tf.linalg.lu.
# Note: PyTorch returns (LU, pivots_indices), while TF returns (lu, permutation_matrix).
tf.linalg.lu(t)
```