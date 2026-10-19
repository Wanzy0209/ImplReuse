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
# Conversion: torch.randn -> tf.random.normal
x = tf.random.normal((2, 2))

# Conversion: torch.compile -> tf.function
# Note: fullgraph=True is the default behavior for tf.function (tracing the whole function)
compiled = tf.function(forward)
```