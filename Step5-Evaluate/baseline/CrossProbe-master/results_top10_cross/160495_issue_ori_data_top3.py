```python
import tensorflow as tf

# from torch._inductor import config
# config.fallback_random = True
# Conversion: PyTorch inductor specific config, not applicable in TensorFlow

# torch.set_grad_enabled(False)
# Conversion: TensorFlow does not track gradients automatically (requires GradientTape)

# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Conversion: tf.function handles dynamic shapes automatically

def eager_fn(x, y):
    return x + y

# @torch.compile()
@tf.function
def fn(x, y):
    return x + y

# x = torch.randn(1, dtype=torch.complex64)
x = tf.random.normal((1,), dtype=tf.complex64)

# y = torch.empty((), dtype=torch.complex64)
y = tf.empty((), dtype=tf.complex64)

eager_fn(x, y)
print("eager success")
fn(x, y)
print("compiler success")
```