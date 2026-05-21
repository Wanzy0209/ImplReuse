import torch
import tensorflow as tf
from tensorflow.python.framework.composite_tensor import CompositeTensor
from tensorflow.python.framework import type_spec
import tensorflow.python.util.nest as nest

# 1. Define a custom class similar to 'Bar' in the original issue.
# It inherits from CompositeTensor to interact with TF's structure handling (similar to pytree).
class MyCompositeTensor(CompositeTensor):
    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def _type_spec(self):
        return MyTensorSpec(self._tensor.shape)

# 2. Define the TypeSpec for the custom class.
# This is analogous to registering the node in pytree.
class MyTensorSpec(type_spec.TypeSpec):
    def __init__(self, shape):
        self._shape = shape

    def value_type(self):
        return MyCompositeTensor

    def _to_components(self, value):
        return [value._tensor]

    def _from_components(self, components):
        return MyCompositeTensor(components[0])

    def _component_specs(self):
        return [tf.TensorSpec(self._shape, tf.float32)]

    def _serialize(self):
        return (self._shape,)

    # Mirroring the 'Bar' class definition from the bug report:
    # def __eq__(self, other): return super().__eq__(other)
    # def __hash__(self): return 0
    # Note: In TF, TypeSpecs are used for guards/tracing keys.
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# 3. Define a container class similar to 'Foo'.
class Container:
    pass

# 4. Define the function to be traced/compiled.
# @tf.function is the TensorFlow equivalent of torch.compile.
@tf.function
def fn(x, obj):
    # Replicate the logic: obj.attr = {3: Bar()}
    # We assign a dictionary containing the custom CompositeTensor to the object's attribute.
    # This tests the tracer's ability to handle side-effects involving custom types.
    obj.attr = {3: MyCompositeTensor(x)}
    return x + 1

# 5. Execution
if __name__ == "__main__":
    c = Container()
    t = tf.ones([3])
    
    # This call triggers the tracing logic. 
    # If TF's tracer has issues similar to PyTorch Dynamo's guard generation 
    # on temporary variables involving custom types with specific hash/eq logic,
    # it might fail here.
    result = fn(t, c)
    
    # Verify execution
    print("Result:", result)
    assert result.shape == (3,)
    
    # Verify the side effect
    assert 3 in c.attr
    assert isinstance(c.attr[3], MyCompositeTensor)
    print("Test passed.")