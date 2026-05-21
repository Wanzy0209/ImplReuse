import torch
import tensorflow as tf
import numpy as np

class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Note: TensorFlow does not have a direct equivalent to pytree.register_constant.
# We use tf.function to mimic the compilation/tracing aspect of torch.compile.

@tf.function
def fn(x, obj):
    # Use the similar API: tf.keras.backend.name_scope
    # This acts as the context manager for the operations, similar to how 
    # torch.compile manages the execution context.
    with tf.keras.backend.name_scope("dynamo_like_scope"):
        # Core logic from the original bug report
        obj.attr = {3: Bar()}
        return x + 1

# Test execution
try:
    input_tensor = tf.ones(3)
    foo_instance = Foo()
    
    result = fn(input_tensor, foo_instance)
    
    # Verify the computation result
    assert tf.reduce_all(result == tf.constant([2.0, 2.0, 2.0])).numpy()
    
    # Verify the side effect on the object
    assert hasattr(foo_instance, 'attr')
    assert 3 in foo_instance.attr
    assert isinstance(foo_instance.attr[3], Bar)
    
    print("Test passed: tf.keras.backend.name_scope handled the logic correctly.")

except Exception as e:
    print(f"Test failed with error: {e}")