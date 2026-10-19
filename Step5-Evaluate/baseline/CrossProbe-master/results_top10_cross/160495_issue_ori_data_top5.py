```python
import tensorflow as tf

# Conversion: PyTorch internal configs (inductor, dynamo) have no direct equivalent in TensorFlow.
# These are specific to PyTorch 2.0+ compilation stack.
# from torch._inductor import config
# config.fallback_random = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Conversion: torch.set_grad_enabled(False)
# TensorFlow 2.x is eager by default. Gradients are tracked via tf.GradientTape.
# Tensors created via ops (like tf.random.normal) are not trainable variables,
# so explicit global gradient disabling is not required here.
# torch.set_grad_enabled(False)

def eager_fn(x, y):
    return x + y

# Conversion: @torch.compile() -> @tf.function
# tf.function compiles the Python function into a static TensorFlow graph.
@tf.function
def fn(x, y):
    return x + y

# Conversion: torch.randn(1, dtype=torch.complex64) -> tf.random.normal
x = tf.random.normal(shape=(1,), dtype=tf.complex64)

# Conversion: torch.empty((), dtype=torch.complex64) -> tf.zeros
# TensorFlow does not support uninitialized memory allocation (torch.empty).
# Using tf.zeros as the standard placeholder for allocated memory.
y = tf.zeros((), dtype=tf.complex64)

eager_fn(x, y)
print("eager success")
fn(x, y)
print("compiler success")
```