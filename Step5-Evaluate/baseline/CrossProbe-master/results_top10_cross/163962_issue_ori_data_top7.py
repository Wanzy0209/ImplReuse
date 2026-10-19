```python
import tensorflow as tf
import tensorly as tl
from tensorly.decomposition import parafac

# Conversion: Set backend to tensorflow instead of pytorch
tl.set_backend("tensorflow")

# Conversion: torch.ones -> tf.ones
# Note: .to("mps") is PyTorch specific for Apple Silicon GPU. 
# TensorFlow handles device placement differently (e.g., with tf.device).
x = tf.ones((12, 3, 12))
print(x.shape)

# Conversion: .detach() is not required in TensorFlow eager execution
weights, factors = parafac(x, 12, init="random", tol=1e-6)

a, m, b = factors
```