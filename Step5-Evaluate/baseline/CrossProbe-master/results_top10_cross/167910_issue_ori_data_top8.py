```python
import tensorflow as tf

# Conversion: nn.Module -> tf.Module
class JITCrash(tf.Module):
    def __init__(self):
        super().__init__()

    # Conversion: forward -> __call__
    def __call__(self, x):
        return x + 1

model = JITCrash()
# Conversion: torch.jit.script -> tf.function
# Note: tf.function creates a callable graph from the model, similar to TorchScript
scripted = tf.function(model)
```