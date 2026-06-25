```python
import tensorflow as tf
from tensorflow.python.framework.func_graph import func_graph_from_py_func

class AnyDimsModelEmpty(tf.keras.Model):
    def __init__(self):
        super(AnyDimsModelEmpty, self).__init__()

    def call(self, x):
        print('input shape:', x.shape)
        # Conversion: torch.ops.aten.any.dims(x, [], False) -> tf.reduce_any(x)
        # In PyTorch ATen, an empty list of dims implies reducing over all dimensions.
        y = tf.reduce_any(x, keepdims=False)
        print('output shape:', y.shape)
        return y

class AnyDimsModelNull(tf.keras.Model):
    def __init__(self):
        super(AnyDimsModelNull, self).__init__()

    def call(self, x):
        print('input shape:', x.shape)
        # Conversion: torch.ops.aten.any.dims(x, None, False) -> tf.reduce_any(x)
        # None implies reducing over all dimensions.
        y = tf.reduce_any(x, keepdims=False)
        print('output shape:', y.shape)
        return y

def process(model, x):
    print('model:', model)
    print('running eager mode...')
    model(x)
    print('exporting...')
    # Conversion: torch.export.export -> func_graph_from_py_func
    # We pass the model's call method and the input arguments to trace the graph.
    func_graph_from_py_func(
        name="exported_func",
        python_func=model.call,
        args=(x,),
        kwargs=None,
        signature=None
    )
    print()
```