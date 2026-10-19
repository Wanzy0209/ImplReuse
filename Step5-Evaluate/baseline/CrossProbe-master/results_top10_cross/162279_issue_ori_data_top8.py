```python
import tensorflow as tf

class AnyDimsModelEmpty(tf.Module):
    def __init__(self):
        super(AnyDimsModelEmpty, self).__init__()

    def __call__(self, x):
        print('input shape:', x.shape)
        # Conversion: torch.ops.aten.any.dims(x, [], False) -> tf.reduce_any(x, axis=[], keepdims=False)
        y = tf.reduce_any(x, axis=[], keepdims=False)
        print('output shape:', y.shape)
        return y

class AnyDimsModelNull(tf.Module):
    def __init__(self):
        super(AnyDimsModelNull, self).__init__()

    def __call__(self, x):
        print('input shape:', x.shape)
        # Conversion: torch.ops.aten.any.dims(x, None, False) -> tf.reduce_any(x, axis=None, keepdims=False)
        y = tf.reduce_any(x, axis=None, keepdims=False)
        print('output shape:', y.shape)
        return y

def process(model, x):
    print('model:', model)
    print('running eager mode...')
    model(x)
    print('exporting...')
    # Conversion: torch.export.export -> tf.function tracing
    # get_concrete_function traces the graph similar to export
    tf.function(model).get_concrete_function(x)
    print()
```