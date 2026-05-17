import tensorflow as tf
import tensorflow.experimental.dtensor as dtensor
from typing import NamedTuple

# Setup minimal mesh for copy_to_mesh
# Note: DTensor requires a mesh to be defined for layout operations
mesh = dtensor.create_mesh([("x", 1)], devices=["CPU:0"])
layout = dtensor.Layout([dtensor.UNSHARDED], mesh)

class MyNamedTuple(NamedTuple):
    first: tf.Tensor
    second: tf.Tensor

class MyNamedTupleSubclass(MyNamedTuple):
    pass

def fn(tup: MyNamedTuple) -> MyNamedTuple:
    extra_info = tf.constant(4.0)
    tup.extra_info = extra_info  # Add dynamic attribute
    
    # Adaptation: Apply copy_to_mesh to the tensors inside the NamedTuple
    # This tests if the transformation process preserves the container's dynamic attributes
    tup.first = dtensor.copy_to_mesh(tup.first, layout)
    tup.second = dtensor.copy_to_mesh(tup.second, layout)
    
    return tup

print("\nTesting NamedTuple with copy_to_mesh:")
extended_tup = MyNamedTupleSubclass(first=tf.constant([2.0]), second=tf.constant(1.0))
result = fn(extended_tup)

try:
    print(f"NamedTuple attribute result: {result.extra_info}")
    assert hasattr(result, 'extra_info'), "Dynamic attribute was lost after copy_to_mesh operation!"
except AttributeError as e:
    print(f"AttributeError: {e}")