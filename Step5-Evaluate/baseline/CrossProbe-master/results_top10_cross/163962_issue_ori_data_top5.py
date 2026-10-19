```python
import tensorflow as tf
import tensorly as tl
# Conversion: Set backend to tensorflow
tl.set_backend("tensorflow")
from tensorly.decomposition import parafac

# Conversion: torch.ones -> tf.ones
# Note: .to("mps") is PyTorch specific for device placement. 
# In TensorFlow, device placement is handled via tf.device context or automatically.
x = tf.ones((12, 3, 12))
print(x.shape)

# Conversion: x.detach() is not required for TensorFlow tensors in this context
weights, factors = parafac(x, 12, init="random", tol=1e-6)

a, m, b = factors
```