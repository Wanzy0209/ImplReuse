```python
import tensorflow as tf

# Note: torch.nn is not strictly needed for this snippet, 
# so we rely on core tensorflow operations.

class Config:
    def __repr__(self):
        return "Config()"


def forward(x, config):
    # Calling repr() on non-constant user object
    # This triggers the bug without the fix
    # Conversion note: In tf.function, non-Tensor arguments (like config) are treated as static.
    # The repr() call executes during the tracing phase.
    return x * len(repr(config))


config = Config()
# Conversion note: torch.randn generates random numbers from a normal distribution.
# tf.random.normal is the equivalent.
x = tf.random.normal((2, 2))

# Conversion note: torch.compile compiles the function into a graph.
# tf.function is the TensorFlow equivalent for graph compilation/tracing.
# fullgraph=True in PyTorch attempts to capture the whole function in one graph,
# which is the default behavior of tf.function.
compiled = tf.function(forward)
```