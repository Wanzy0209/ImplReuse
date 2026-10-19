```python
import tensorflow as tf

# config.fallback_random = True
# PyTorch inductor specific config, not applicable in TensorFlow.

# torch.set_grad_enabled(False)
# TensorFlow uses tf.GradientTape; gradients are not calculated by default in eager mode.

# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# PyTorch dynamo specific config, not applicable in TensorFlow.

def eager_fn(x, y):
    return x + y

@tf.function
def fn(x, y):
    return x + y

# torch.randn(1, dtype=torch.complex64) -> tf.random.normal
x = tf.random.normal((1,), dtype=tf.complex64)

# torch.empty((), dtype=torch.complex64) -> tf.zeros
# TensorFlow does not support uninitialized tensors in the public API; using zeros as a safe equivalent.
y = tf.zeros((), dtype=tf.complex64)

eager_fn(x, y)
print("eager success")
fn(x, y)
print("compiler success")
```