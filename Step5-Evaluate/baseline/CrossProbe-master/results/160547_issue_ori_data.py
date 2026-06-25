```python
import tensorflow as tf
from collections import namedtuple
from tensorflow.python.framework.func_graph import func_graph_from_py_func

def test_namedtuple(self):
    Point = namedtuple('Point', 'x y')
    
    # Conversion: torch.nn.Module -> tf.Module
    class M(tf.Module):
        # Conversion: forward -> __call__ to make the instance callable
        def __call__(self, x, y):
            return x + y 
    
    # Conversion: torch.ones -> tf.ones
    inp = Point(tf.ones(3), tf.ones(3))
    print(M()(*inp))
    
    # errors
    # Conversion: torch.export.export -> func_graph_from_py_func
    # Note: func_graph_from_py_func does not have a 'strict' parameter.
    ep = func_graph_from_py_func(M(), args=inp, kwargs=None, name='export_strict_false')
    print(ep)

    # succeeds
    ep = func_graph_from_py_func(M(), args=inp, kwargs=None, name='export_strict_true')
    print(ep)

    # workaround could be to convert namedtuple to a kwarg
    inp_kwargs = {field: getattr(inp, field) for field in inp._fields}
    ep = func_graph_from_py_func(M(), args=(), kwargs=inp_kwargs, name='export_kwargs')
    print(ep)
```