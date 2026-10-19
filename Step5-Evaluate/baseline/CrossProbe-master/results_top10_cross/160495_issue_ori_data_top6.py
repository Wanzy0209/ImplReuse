```python
import tensorflow as tf
# import torch.nn as nn
# import torch.nn.functional as F
# from torch._inductor import config

# config.fallback_random = True
# Note: PyTorch internal configs have no direct TensorFlow equivalent.

# torch.set_grad_enabled(False)
# Note: In TensorFlow, gradients are only tracked within tf.GradientTape.
# Explicit global disabling is not required for inference.

# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: No direct equivalent in TensorFlow XLA/graph compilation.

def eager_fn(x, y):
    return x + y

# @torch.compile() translates to @tf.function(jit_compile=True) for XLA compilation
@tf.function(jit_compile=True)
def fn(x, y):
    return x + y

# torch.randn(1, dtype=torch.complex64) -> tf.random.normal
x = tf.random.normal((1,), dtype=tf.complex64)

# torch.empty((), dtype=torch.complex64) -> tf.empty
y = tf.empty((), dtype=tf.complex64)

eager_fn(x, y)
print("eager success")
fn(x, y)
print("compiler success")
```