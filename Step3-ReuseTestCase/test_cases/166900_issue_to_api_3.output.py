import tensorflow as tf

# Preserving the class definitions from the original bug reproduction logic
class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Note: TensorFlow does not use pytree.register_constant in the same way as PyTorch,
# but we preserve the structure of the test case to reflect the code similarity.

# Function that mimics the original 'fn' structure
# but leverages the similar API (tf.config.list_logical_devices)
def fn(x, obj):
    # Original logic involved setting an attribute on the object.
    # Here we use the similar API to retrieve device information.
    # We pass 'None' to list all logical devices, similar to the default behavior.
    devices = tf.config.list_logical_devices(device_type=None)
    
    # Mimic the side effect from the original bug
    obj.attr = devices
    
    # Return a value to maintain the function signature similarity
    return x + 1

# Test execution
# In the original, x is a tensor. We use a tf.Tensor here.
x = tf.ones(3)
obj = Foo()

# Call the function
result = fn(x, obj)

# Assertions to verify the behavior and ensure the test is valid
assert hasattr(obj, 'attr'), "Object attribute 'attr' was not set"
assert isinstance(obj.attr, list), "Attribute 'attr' should be a list of devices"
assert result.shape == (3,), "Result shape mismatch"

print("Test passed successfully.")