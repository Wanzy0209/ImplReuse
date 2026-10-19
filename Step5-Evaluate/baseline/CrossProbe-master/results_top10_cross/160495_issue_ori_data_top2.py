```python
import tensorflow as tf

# PyTorch: config.fallback_random = True
# TensorFlow: No direct equivalent for inductor configuration.

# PyTorch: torch.set_grad_enabled(False)
# TensorFlow: Gradients are only calculated when using tf.GradientTape.
# No global flag is required to disable gradients for eager execution.

# PyTorch: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# TensorFlow: tf.function handles dynamic shapes by default via re-tracing.

def eager_fn(x, y):
    return x + y

# PyTorch: @torch.compile()
# TensorFlow: @tf.function is the equivalent decorator for graph compilation/optimization.
@tf.function
def fn(x, y):
    return x + y

# PyTorch: x = torch.randn(1, dtype=torch.complex64)
# TensorFlow: Use tf.random.normal to generate random numbers.
x = tf.random.normal(shape=(1,), dtype=tf.complex64)

# PyTorch: y = torch.empty((), dtype=torch.complex64)
# TensorFlow: Use tf.empty to allocate uninitialized memory.
y = tf.empty(shape=(), dtype=tf.complex64)

eager_fn(x, y)
print("eager success")
fn(x, y)
print("compiler success")
```