```python
import tensorflow as tf

class MyModule(tf.keras.Model):  # Conversion: torch.nn.Module -> tf.keras.Model
    def __init__(self, param: int = 42) -> None:
        super().__init__()  # Conversion: Required to initialize the base class
        self.some_param = param  # <- raises an `unresolved-attribute` error
```