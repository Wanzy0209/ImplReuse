```python
import tensorflow as tf


class Config:
    def __repr__(self):
        return "Config()"


def forward(x, config):
    # Calling repr() on non-constant user object
    # This triggers the bug without the fix
    return x * len(repr(config))


config = Config()
# Conversion: torch.randn generates random numbers from a normal distribution
x = tf.random.normal((2, 2))

# Conversion: torch.compile optimizes the function into a graph, similar to tf.function
compiled = tf.function(forward)
```