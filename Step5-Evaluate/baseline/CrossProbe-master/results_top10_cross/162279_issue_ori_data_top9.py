```python
import tensorflow as tf

class AnyDimsModelEmpty(tf.keras.Model):
    def __init__(self):
        super(AnyDimsModelEmpty, self).__init__()

    def call(self, x):
        print('input shape:', x.shape)
        # Conversion: torch.ops.aten.any.dims(x, [], False) -> tf.reduce_any with axis=[] (identity)
        y = tf.reduce_any(x, axis=[], keepdims=False)
        print('output shape:', y.shape)
        return y

class AnyDimsModelNull(tf.keras.Model):
    def __init__(self):
        super(AnyDimsModelNull, self).__init__()

    def call(self, x):
        print('input shape:', x.shape)
        # Conversion: torch.ops.aten.any.dims(x, None, False) -> tf.reduce_any with axis=None (reduce all)
        y = tf.reduce_any(x, axis=None, keepdims=False)
        print('output shape:', y.shape)
        return y

def process(model, x):
    print('model:', model)
    print('running eager mode...')
    model(x)
    print('exporting...')
    # Conversion: torch.export.export -> tf.saved_model.save
    tf.saved_model.save(model, "exported_model")
    print()
```