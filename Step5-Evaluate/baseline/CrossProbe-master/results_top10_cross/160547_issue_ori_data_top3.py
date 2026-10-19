```python
import tensorflow as tf
from collections import namedtuple

def test_namedtuple(self):
    Point = namedtuple('Point', 'x y')
    
    class M(tf.Module):
        # Conversion: PyTorch 'forward' -> TensorFlow '__call__'
        def __call__(self, x, y):
            return x + y 
    
    # Conversion: torch.ones -> tf.ones
    inp = Point(tf.ones(3), tf.ones(3))
    print(M()(*inp))
    
    # errors
    # Conversion: torch.export.export -> tf.function + get_concrete_function
    # Note: TensorFlow does not have a 'strict' parameter for graph tracing.
    # We simulate the export process by obtaining a concrete function.
    m = M()
    m_tf = tf.function(m)
    
    # In PyTorch, the namedtuple 'inp' is passed as the args tuple.
    # In TensorFlow, we unpack it to match the function signature.
    # Note: The original PyTorch code comments this as 'errors', but in TF
    # this operation typically succeeds.
    try:
        ep = m_tf.get_concrete_function(*inp)
        print(ep)
    except Exception as e:
        print(e)

    # succeeds
    try:
        ep = m_tf.get_concrete_function(*inp)
        print(ep)
    except Exception as e:
        print(e)

    # workaround could be to convert namedtuple to a kwarg
    inp_kwargs = {field: getattr(inp, field) for field in inp._fields}
    # Conversion: export with kwargs -> get_concrete_function with kwargs
    ep = m_tf.get_concrete_function(**inp_kwargs)
    print(ep)
```