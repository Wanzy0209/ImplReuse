```python
import tensorflow as tf
import tensorly as tl
# Set backend to tensorflow instead of pytorch
tl.set_backend("tensorflow")
from tensorly.decomposition import parafac

# torch.ones(12,3,12) -> tf.ones((12, 3, 12))
# .to("mps") is PyTorch specific for Apple Silicon GPU. 
# In TensorFlow, device placement is handled via tf.device context or global config.
x = tf.ones((12, 3, 12))
print(x.shape)

# x.detach() is not required in TensorFlow eager execution as tensors do not track history by default.
weights, factors = parafac(x, 12, init="random", tol=1e-6)

a, m, b = factors
```