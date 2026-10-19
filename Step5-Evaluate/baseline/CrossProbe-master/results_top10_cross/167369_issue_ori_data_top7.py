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
# Conversion: torch.randn(2, 2) -> tf.random.normal((2, 2))
x = tf.random.normal((2, 2))

# Conversion: torch.compile(forward, fullgraph=True) -> tf.function(forward)
compiled = tf.function(forward)
```