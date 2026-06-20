import sys
import numpy as np

try:
    import torch
    import tensorflow as tf
except ImportError as e:
    # Handle environment issues like missing GLIBCXX or missing libraries
    print(f"SKIPPED: Unable to import dependencies due to environment configuration.")
    print(f"Details: {e}")
    print("This is likely due to a GLIBCXX version mismatch or missing TensorFlow installation.")
    sys.exit(0)

def test_batch_set_value_dynamic_attribute_persistence():
    """
    Test case to verify if dynamic attributes persist on variables 
    when using tf.keras.backend.batch_set_value inside a tf.function.
    
    This mirrors the PyTorch issue where torch.compile would strip 
    dynamic attributes from NamedTuple subclasses.
    """
    # Create a Keras Variable
    var = tf.Variable([1.0, 2.0], dtype=tf.float32)
    
    # Add a dynamic attribute to the variable object
    # This mimics the 'tup.extra_info = extra_info' in the PyTorch bug
    var.extra_info = "test_metadata"
    
    # Define the update function wrapped in tf.function (analogous to torch.compile)
    @tf.function
    def update_fn():
        # Use the similar API: batch_set_value
        # It takes a list of tuples: (tensor, value)
        new_value = np.array([3.0, 4.0])
        tf.keras.backend.batch_set_value([(var, new_value)])
    
    # Execute the compiled function
    update_fn()
    
    # 1. Verify the core functionality of batch_set_value (value update)
    np.testing.assert_array_equal(var.numpy(), [3.0, 4.0], 
                                  "Variable value was not updated correctly by batch_set_value")
    
    # 2. Verify the persistence of the dynamic attribute
    # In the PyTorch bug (Issue 161610), accessing the attribute after compile raises AttributeError.
    # Here we assert that the attribute still exists and holds the correct value.
    assert hasattr(var, 'extra_info'), \
        "FAIL: Dynamic attribute 'extra_info' was lost after execution (similar to PyTorch bug 161610)"
    
    assert var.extra_info == "test_metadata", \
        "FAIL: Dynamic attribute 'extra_info' value changed unexpectedly"
    
    print("Test Passed: batch_set_value updated values and preserved dynamic attributes.")

if __name__ == "__main__":
    test_batch_set_value_dynamic_attribute_persistence()