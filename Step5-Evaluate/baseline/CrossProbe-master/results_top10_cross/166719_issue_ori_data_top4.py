```python
import tensorflow as tf

# Conversion: torch.nn.Module maps to tf.keras.layers.Layer
class MyModule(tf.keras.layers.Layer):
    def __init__(self, param: int = 42) -> None:
        # Conversion: super().__init__() is required in Keras to initialize the base Layer
        # This resolves the 'unresolved-attribute' error mentioned in the source
        super(MyModule, self).__init__()
        self.some_param = param  # <- raises an `unresolved-attribute` error
```