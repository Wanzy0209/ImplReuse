```python
import tensorflow as tf
from collections import namedtuple

def test_namedtuple(self):
    Point = namedtuple('Point', 'x y')
    
    class M(tf.Module):
        # Conversion: Use tf.Module and @tf.function for graph definition
        @tf.function
        def __call__(self, x, y):
            return x + y 
    
    # Conversion: torch.ones -> tf.ones
    inp = Point(tf.ones(3), tf.ones(3))
    print(M()(*inp))
    
    # errors
    # Conversion: torch.export.export -> tf.function.get_concrete_function
    # Note: TF does not have a 'strict' parameter for export. 
    # We trace the function to get the exported graph representation.
    concrete_fn = M().__call__.get_concrete_function(*inp)
    print(concrete_fn)

    # succeeds
    concrete_fn = M().__call__.get_concrete_function(*inp)
    print(concrete_fn)

    # workaround could be to convert namedtuple to a kwarg
    inp_kwargs = {field: getattr(inp, field) for field in inp._fields}
    # Conversion: Exporting with keyword arguments
    concrete_fn = M().__call__.get_concrete_function(**inp_kwargs)
    print(concrete_fn)
```