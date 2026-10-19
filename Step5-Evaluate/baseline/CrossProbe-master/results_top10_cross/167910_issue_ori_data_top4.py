```python
import tensorflow as tf

# Conversion: nn.Module -> tf.keras.Model
class JITCrash(tf.keras.Model):
    def __init__(self):
        super(JITCrash, self).__init__()

    # Conversion: forward -> call
    def call(self, x):
        return x + 1

model = JITCrash()
# Conversion: torch.jit.script -> tf.function
# tf.function creates a static graph representation of the model
scripted = tf.function(model)
```