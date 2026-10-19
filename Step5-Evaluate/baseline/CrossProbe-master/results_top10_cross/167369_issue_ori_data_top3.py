```python
import tensorflow as tf

# Note: torch.nn is not used in the snippet, so it is omitted in the translation

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

# Conversion: torch.compile -> tf.function
# fullgraph=True in PyTorch attempts to capture the whole model into a single graph.
# tf.function does this by default (tracing the whole function).
compiled = tf.function(forward)
```