```python
import tensorflow as tf
import tensorly as tl
# Set backend to tensorflow
tl.set_backend("tensorflow")
from tensorly.decomposition import parafac

# Create a tensor of ones. Note: TensorFlow uses a tuple for the shape argument.
# The .to("mps") device placement is handled by TensorFlow's runtime or tf.device context.
x = tf.ones((12, 3, 12))
print(x.shape)

# Perform parafac decomposition.
# x.detach() is not required for TF tensors created directly, 
# but tf.stop_gradient() would be the equivalent if detaching from a graph was needed.
weights, factors = parafac(x, 12, init="random", tol=1e-6)

a, m, b = factors
```