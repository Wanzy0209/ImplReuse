```python
import tensorflow as tf

class MyModule(tf.keras.Model):  # Converted from torch.nn.Module
    def __init__(self, param: int = 42) -> None:
        super().__init__()  # Required for proper initialization in Keras
        self.some_param = param  # <- raises an `unresolved-attribute` error
```