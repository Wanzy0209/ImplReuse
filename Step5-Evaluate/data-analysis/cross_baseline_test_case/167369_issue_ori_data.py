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
# fullgraph=True implies strict graph capturing, which is the default behavior of tf.function
compiled = tf.function(forward)
```