```python
import tensorflow as tf


class Config:
    def __repr__(self):
        return "Config()"


def forward(x, config):
    # Calling repr() on non-constant user object
    # This triggers the bug without the fix
    # Conversion: In tf.function, len(repr(config)) is evaluated at trace time
    return x * len(repr(config))


config = Config()
# Conversion: torch.randn(2, 2) creates a random tensor from a normal distribution
x = tf.random.normal((2, 2))

# Conversion: torch.compile compiles the function into a graph.
# In TensorFlow, tf.function is used to create a graph-compatible callable.
# fullgraph=True in PyTorch implies strict graph compilation, similar to tf.function's default behavior.
compiled = tf.function(forward)
```