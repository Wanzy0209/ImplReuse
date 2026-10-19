```python
import tensorflow as tf

# PyTorch specific internal configs, no direct equivalent in TensorFlow
# from torch._inductor import config
# config.fallback_random = True

# TensorFlow tracks gradients only within tf.GradientTape.
# No global gradient disable flag is needed for eager execution.
# torch.set_grad_enabled(False)

# PyTorch dynamo config, no direct equivalent in TensorFlow
# torch._dynamo.config.capture_dynamic_output_shape_ops = True

def eager_fn(x, y):
    return x + y

# @torch.compile() translates to @tf.function in TensorFlow
@tf.function
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