```python
import tensorflow as tf

class MyModule(tf.keras.layers.Layer):  # Conversion: torch.nn.Module -> tf.keras.layers.Layer
    def __init__(self, param: int = 42) -> None:
        super().__init__()  # Conversion: Added super().__init__() call required by Keras
        self.some_param = param  # <- raises an `unresolved-attribute` error
```