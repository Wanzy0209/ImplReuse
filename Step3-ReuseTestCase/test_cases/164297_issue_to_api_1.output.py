import tensorflow as tf
from tensorflow.experimental import ExtensionType, dispatch_for_unary_elementwise_apis

# Define a custom type to test the dispatch mechanism, 
# analogous to the custom type involved in the original bug.
class CustomTensor(ExtensionType):
    values: tf.Tensor

# Register a handler for this custom type using the similar API.
# This tests the binding and registration logic that was failing in the original bug.
@dispatch_for_unary_elementwise_apis(CustomTensor)
def custom_unary_handler(api_func, x):
    # Apply the elementwise operation to the underlying tensor
    return CustomTensor(api_func(x.values))

def test_dispatch_for_unary_elementwise_apis():
    """
    Test case to verify that custom types can be registered and used
    with unary elementwise APIs without causing segmentation faults or
    binding errors, reflecting the stability concerns of the original issue.
    """
    # Create an instance of the custom type
    x = CustomTensor(tf.constant([-1.0, -2.0, 3.0]))
    
    # Perform a unary operation (tf.abs) which triggers the dispatcher.
    # In the original bug, accessing/converting the type caused a segfault.
    # Here we verify the dispatch mechanism handles the type correctly.
    y = tf.abs(x)
    
    # Assertions to verify correct behavior
    assert isinstance(y, CustomTensor)
    assert tf.reduce_all(y.values == tf.constant([1.0, 2.0, 3.0]))

if __name__ == "__main__":
    test_dispatch_for_unary_elementwise_apis()
    print("Test passed successfully.")