import torch
import tensorflow as tf
import numpy as np

# Replicating the class structure from the original issue to maintain code similarity
class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Setup: Create a variable to hold state (analogous to obj.attr in the original issue)
# We initialize it with zeros.
var = tf.Variable(np.zeros((3,)))

# The compiled function (analogous to @torch.compile)
# tf.function is the TensorFlow equivalent for graph compilation/tracing
@tf.function
def fn(x, obj):
    # Original logic: obj.attr = {3: Bar()}
    # Adapted logic: Use tf.keras.backend.set_value to update the variable.
    # This mirrors the state update pattern of the original bug.
    # Note: set_value expects a numpy array or tensor, so we pass 'x'.
    tf.keras.backend.set_value(var, x)
    return var + 1

# Test execution
input_data = np.ones((3,))
# We pass an instance of Foo to match the original signature, though it's unused in the body
result = fn(input_data, Foo())

# Assertion: Verify the variable was updated correctly inside the compiled context
# This confirms the state update worked, addressing the core pattern of the original issue
assert np.array_equal(tf.keras.backend.get_value(var), input_data)
print("Test passed: set_value works inside tf.function")