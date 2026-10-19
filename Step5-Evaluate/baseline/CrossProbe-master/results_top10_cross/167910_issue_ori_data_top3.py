```python
import tensorflow as tf

class JITCrash(tf.Module):
    def __init__(self):
        super().__init__()

    # Conversion: PyTorch uses 'forward' for the computation logic.
    # In TensorFlow (tf.Module), this is typically defined in '__call__'.
    def __call__(self, x):
        return x + 1

model = JITCrash()

# Conversion: torch.jit.script converts the model into a static graph.
# In TensorFlow, tf.function creates a graph (AutoGraph) from the Python code.
# Applying tf.function to the model instance returns a callable graph version.
scripted = tf.function(model)
```