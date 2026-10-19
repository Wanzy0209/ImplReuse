```python
import tensorflow as tf
import tensorly as tl
from tensorly.decomposition import parafac

# Set TensorLy backend to TensorFlow
tl.set_backend("tensorflow")

# Create a tensor of ones. 
# Note: .to("mps") is PyTorch specific for Apple Silicon GPU acceleration.
# In TensorFlow, device placement is handled automatically or via tf.device contexts.
x = tf.ones((12, 3, 12))
print(x.shape)

# Perform CP decomposition. 
# x.detach() is not required in TensorFlow for constant tensors.
weights, factors = parafac(x, 12, init="random", tol=1e-6)

a, m, b = factors
```