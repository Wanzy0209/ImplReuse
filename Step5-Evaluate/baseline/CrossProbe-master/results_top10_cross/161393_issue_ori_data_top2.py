```python
import tensorflow as tf

# PyTorch specific compiler configurations, not applicable in TensorFlow
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True

def f(x):
    # PyTorch: x.nonzero() returns indices of non-zero elements
    # TensorFlow: tf.where(x != 0) returns indices where the condition is True
    nz = tf.where(x != 0)
    return nz[:-1]

# PyTorch: torch.compile(f, fullgraph=True) compiles the function into a graph
# TensorFlow: tf.function(f) compiles the function into a TensorFlow graph
out = tf.function(f)(tf.random.normal((3, 4)))
print(out)
```