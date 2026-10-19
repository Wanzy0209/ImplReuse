```python
import tensorflow as tf
import tensorly as tl
# Set backend to TensorFlow
tl.set_backend("tensorflow")
from tensorly.decomposition import parafac

# Create a tensor of ones. 
# Note: PyTorch's .to("mps") moves data to a specific device. 
# In TensorFlow, device placement is typically handled via 'with tf.device(...):' context managers.
# Here we create the tensor on the default device.
x = tf.ones((12, 3, 12))
print(x.shape)

# Perform PARAFAC decomposition.
# In PyTorch, x.detach() is used to remove the tensor from the computational graph.
# In TensorFlow, tf.ones creates a constant tensor which does not track gradients, 
# so no explicit detach is needed.
weights, factors = parafac(x, 12, init="random", tol=1e-6)

a, m, b = factors
```