```python
import tensorflow as tf
import tensorly as tl
# Set TensorLy backend to TensorFlow
tl.set_backend("tensorflow")
from tensorly.decomposition import parafac

# Create a tensor of ones. Note: PyTorch's .to("mps") is specific to PyTorch device handling.
# In TensorFlow, device placement is typically handled by the runtime or with tf.device.
x = tf.ones((12, 3, 12))
print(x.shape)

# Run PARAFAC decomposition. x.detach() is not required in TensorFlow eager execution
# as tensors do not track gradients automatically.
weights, factors = parafac(x, 12, init="random", tol=1e-6)

a, m, b = factors
```