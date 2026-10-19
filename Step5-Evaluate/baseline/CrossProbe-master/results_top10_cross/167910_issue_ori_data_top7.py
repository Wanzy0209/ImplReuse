```python
import tensorflow as tf

# Conversion: torch.nn.Module -> tf.Module
class JITCrash(tf.Module):
    def __init__(self):
        super().__init__()

    # Conversion: forward -> __call__ (standard entry point for TensorFlow modules)
    def __call__(self, x):
        return x + 1

model = JITCrash()
# Conversion: torch.jit.script(model) -> tf.function(model)
# tf.function creates a callable graph from the Python function/module
scripted = tf.function(model)
```