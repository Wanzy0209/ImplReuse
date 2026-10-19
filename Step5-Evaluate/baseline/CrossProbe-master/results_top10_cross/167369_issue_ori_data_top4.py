```python
import tensorflow as tf
# import tensorflow.nn as nn # Not strictly used in this snippet but preserved for structure

class Config:
    def __repr__(self):
        return "Config()"

def forward(x, config):
    # Calling repr() on non-constant user object
    # This triggers the bug without the fix
    # Conversion note: In tf.function, Python operations like repr() are executed
    # during the tracing phase, and the result is baked into the graph as a constant.
    return x * len(repr(config))

config = Config()
# Conversion: torch.randn -> tf.random.normal
x = tf.random.normal((2, 2))

# Conversion: torch.compile -> tf.function
# fullgraph=True is the default behavior of tf.function (attempting to capture the whole graph)
compiled = tf.function(forward)
```