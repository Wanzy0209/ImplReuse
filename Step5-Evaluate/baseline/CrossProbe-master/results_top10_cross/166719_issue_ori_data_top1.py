```python
import tensorflow as tf

class MyModule(tf.keras.layers.Layer):
    def __init__(self, param: int = 42, **kwargs) -> None:
        # Conversion: Call parent constructor with **kwargs for Keras compatibility
        super(MyModule, self).__init__(**kwargs)
        self.some_param = param  # <- raises an `unresolved-attribute` error
```