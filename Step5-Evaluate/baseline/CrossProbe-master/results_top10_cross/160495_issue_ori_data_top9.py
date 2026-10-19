```python
import tensorflow as tf

# PyTorch: from torch._inductor import config
# PyTorch: config.fallback_random = True
# TensorFlow: No direct equivalent for inductor specific flags.
# TensorFlow uses XLA for compilation optimization.

# PyTorch: torch.set_grad_enabled(False)
# TensorFlow: TF 2.x is eager by default. Gradients are only calculated within tf.GradientTape.
# To strictly mimic "no grad", we just don't use a tape.

# PyTorch: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# TensorFlow: tf.function handles dynamic shapes automatically in many cases.

def eager_fn(x, y):
    return x + y

# PyTorch: @torch.compile()
# TensorFlow: @tf.function compiles the function into a static graph.
@tf.function
def fn(x, y):
    return x + y

# PyTorch: x = torch.randn(1, dtype=torch.complex64)
# TensorFlow: tf.random.normal generates random values from a normal distribution.
x = tf.random.normal((1,), dtype=tf.complex64)

# PyTorch: y = torch.empty((), dtype=torch.complex64)
# TensorFlow: tf.empty creates a tensor without initializing entries.
y = tf.empty((), dtype=tf.complex64)

eager_fn(x, y)
print("eager success")
fn(x, y)
print("compiler success")
```