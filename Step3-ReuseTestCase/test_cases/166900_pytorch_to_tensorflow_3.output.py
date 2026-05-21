import tensorflow as tf

# Enable eager execution as requested by the target API
tf.compat.v1.enable_eager_execution()

class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

def fn(x, obj):
    # Replicate the logic from the original bug report:
    # Setting an attribute on the input object with a dictionary containing a custom class instance
    obj.attr = {3: Bar()}
    return x + 1

# Execute the test
try:
    # Create inputs
    input_tensor = tf.ones(3)
    foo_instance = Foo()
    
    # Call the function
    result = fn(input_tensor, foo_instance)
    
    # Verify the tensor operation result
    expected = tf.constant([2.0, 2.0, 2.0])
    assert tf.reduce_all(result == expected).numpy(), "Tensor operation result mismatch"
    
    # Verify the side effect on the object
    assert hasattr(foo_instance, 'attr'), "Attribute 'attr' not set on object"
    assert 3 in foo_instance.attr, "Key 3 not found in obj.attr"
    assert isinstance(foo_instance.attr[3], Bar), "Value in obj.attr is not an instance of Bar"
    
    print("Test passed: TensorFlow eager execution handled the logic correctly.")

except Exception as e:
    print(f"Test failed with error: {e}")
    raise