```python
import tensorly as tl
# Set backend to TensorFlow
tl.set_backend("tensorflow")
from tensorly.decomposition import parafac
import tensorflow as tf

# Create a tensor of ones. Shape is passed as a tuple in TensorFlow.
# Device placement is handled by the TensorFlow runtime configuration.
x = tf.ones((12, 3, 12))
print(x.shape)

# Perform PARAFAC decomposition.
# Note: tf.ones creates a constant, so no detach (stop_gradient) is needed.
weights, factors = parafac(x, 12, init="random", tol=1e-6)

a, m, b = factors
```