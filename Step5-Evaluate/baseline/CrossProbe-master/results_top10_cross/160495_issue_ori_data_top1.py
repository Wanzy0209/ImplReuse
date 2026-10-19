```python
import tensorflow as tf

# PyTorch: from torch._inductor import config; config.fallback_random = True
# PyTorch: torch.set_grad_enabled(False)
# Note: TensorFlow does not have a global gradient switch; gradients are only calculated within tf.GradientTape.

# PyTorch: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: PyTorch specific configuration, ignored in TensorFlow.

def eager_fn(x, y):
    return x + y

# PyTorch: @torch.compile()
# Conversion: Use @tf.function to compile the function into a static graph in TensorFlow
@tf.function
def fn(x, y):
    return x + y

# PyTorch: x = torch.randn(1, dtype=torch.complex64)
# Conversion: Use tf.random.normal to generate random values
x = tf.random.normal((1,), dtype=tf.complex64)

# PyTorch: y = torch.empty((), dtype=torch.complex64)
# Conversion: Use tf.empty to create an uninitialized tensor
y = tf.empty((), dtype=tf.complex64)

eager_fn(x, y)
print("eager success")
fn(x, y)
print("compiler success")
```