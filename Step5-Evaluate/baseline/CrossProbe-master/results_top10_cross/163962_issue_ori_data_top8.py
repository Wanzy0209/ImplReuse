```python
import tensorly as tl
import tensorflow as tf

# Set backend to TensorFlow
tl.set_backend("tensorflow")
from tensorly.decomposition import parafac

# Create a tensor of ones. Note: In TensorFlow, device placement is typically handled 
# via context managers (e.g., with tf.device('/GPU:0')) rather than a .to() method.
x = tf.ones((12, 3, 12))
print(x.shape)

# Perform parafac decomposition. tf.stop_gradient is the conceptual equivalent of .detach(),
# but is not strictly required for constant tensors like tf.ones.
weights, factors = parafac(x, 12, init="random", tol=1e-6)

a, m, b = factors
```