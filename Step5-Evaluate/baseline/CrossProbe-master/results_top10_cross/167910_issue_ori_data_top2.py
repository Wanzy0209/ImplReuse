```python
import tensorflow as tf

# Conversion: nn.Module -> tf.Module
class JITCrash(tf.Module):
    def __init__(self):
        super().__init__()

    # Conversion: torch.jit.script logic -> @tf.function decorator
    # This traces the function and creates a TensorFlow Graph.
    @tf.function
    def forward(self, x):
        return x + 1

model = JITCrash()
# Conversion: torch.jit.script(model) -> tf.function(model.forward)
# This obtains a callable ConcreteFunction (Graph) similar to a ScriptModule.
scripted = tf.function(model.forward)
```