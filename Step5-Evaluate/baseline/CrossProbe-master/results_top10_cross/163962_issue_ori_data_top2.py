```python
import tensorly as tl
import tensorflow as tf

# Set backend to TensorFlow
tl.set_backend("tensorflow")
from tensorly.decomposition import parafac

# Create a tensor of ones. Note: .to("mps") is PyTorch specific for device placement.
# In TensorFlow, device placement is typically handled via context managers (e.g., with tf.device(...)).
x = tf.ones((12, 3, 12))
print(x.shape)

# parafac uses the TensorFlow backend. x is a tf.Tensor, so no .detach() is needed.
weights, factors = parafac(x, 12, init="random", tol=1e-6)

a, m, b = factors
```