```python
import tensorly as tl
import tensorflow as tf

# Set backend to TensorFlow
tl.set_backend("tensorflow")
from tensorly.decomposition import parafac

# Create a tensor of ones. 
# Note: PyTorch's .to("mps") is specific to device placement. 
# In TensorFlow, device placement is handled via 'with tf.device(...):' or system defaults.
x = tf.ones((12, 3, 12))
print(x.shape)

# Perform PARAFAC decomposition.
# x.detach() is removed as it is specific to PyTorch autograd; 
# tf tensors created via tf.ones do not track gradients and do not require detaching.
weights, factors = parafac(x, 12, init="random", tol=1e-6)

a, m, b = factors
```