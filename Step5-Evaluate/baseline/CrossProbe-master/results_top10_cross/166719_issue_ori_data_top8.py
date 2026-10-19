```python
import tensorflow as tf

class MyModule(tf.keras.layers.Layer):
    def __init__(self, param: int = 42) -> None:
        # Conversion: super().__init__() is required in Keras to properly initialize the Layer
        super().__init__()
        self.some_param = param  # <- raises an `unresolved-attribute` error
```