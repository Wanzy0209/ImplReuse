```python
import tensorflow as tf


class Config:
    def __repr__(self):
        return "Config()"


def forward(x, config):
    # Calling repr() on non-constant user object
    # This triggers the bug without the fix
    # Conversion: In TensorFlow, tf.function traces the graph. Python operations
    # like repr() are executed during tracing and treated as constants.
    return x * len(repr(config))


config = Config()
# Conversion: torch.randn creates a tensor with random values from a normal distribution.
# TensorFlow equivalent is tf.random.normal.
x = tf.random.normal((2, 2))

# Conversion: torch.compile compiles a function into a graph.
# TensorFlow equivalent is tf.function.
compiled = tf.function(forward)
```