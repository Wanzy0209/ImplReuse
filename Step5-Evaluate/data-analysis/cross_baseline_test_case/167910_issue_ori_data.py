```python
import tensorflow as tf

class JITCrash(tf.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x + 1

model = JITCrash()
# Conversion: torch.jit.script(model) -> tf.keras.backend.image_data_format()
# Note: The functionality changes from scripting a model to checking the image data format convention.
scripted = tf.keras.backend.image_data_format()
```