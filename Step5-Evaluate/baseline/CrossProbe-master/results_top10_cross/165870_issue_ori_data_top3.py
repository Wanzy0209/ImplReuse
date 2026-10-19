```python
import tensorflow as tf

# Conversion note: PyTorch's device="mps" (Metal Performance Shaders) is specific to Apple Silicon.
# In TensorFlow, device placement is typically handled via context managers (e.g., with tf.device('/GPU:0')).
# We create the tensor here which will use the default device.
t = tf.constant([[1.0, 2.0], [2.0, 4.0]])

# Conversion note: tf.linalg.lu is the TensorFlow equivalent for LU factorization.
# It returns a tuple (lu, p) similar to torch.linalg.lu_factor.
# Note: PyTorch pivots are 1-indexed, while TensorFlow pivots are 0-indexed.
tf.linalg.lu(t)
```