```python
import tensorflow as tf

class MyModule(tf.keras.Model):  # Conversion: torch.nn.Module maps to tf.keras.Model
    def __init__(self, param: int = 42, **kwargs) -> None:  # Conversion: Added **kwargs for Keras base class compatibility
        super().__init__(**kwargs)  # Conversion: Explicit super call required in Keras
        self.some_param = param  # <- raises an `unresolved-attribute` error
```