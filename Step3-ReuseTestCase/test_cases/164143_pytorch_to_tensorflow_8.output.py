import torch
import tensorflow as tf

def test_name_scope_in_compiled_mode():
    """
    Adapted from PyTorch Issue 164143: DebugMode silently disables torch.compile.
    
    This test verifies that tf.keras.name_scope works correctly (does not silently 
    disable or skip) when used inside tf.function, which is the TensorFlow equivalent 
    of torch.compile. The original PyTorch bug involved a dispatch mode causing 
    compilation to skip silently; here we ensure the naming context is preserved 
    during graph compilation.
    """
    
    # Define a function decorated with tf.function (analogous to torch.compile)
    @tf.function
    def compiled_function():
        # Use the similar API: tf.keras.name_scope
        with tf.keras.name_scope("debug_scope"):
            # Create a tensor to check if the scope name is applied
            return tf.constant(1.0, name="test_tensor")

    # Execute the compiled function
    result = compiled_function()

    # Assertion: Verify that the name_scope was active and not "silently disabled"
    # inside the compiled context. The tensor name should include the scope prefix.
    assert "debug_scope" in result.name, (
        f"Expected 'debug_scope' in tensor name, but got '{result.name}'. "
        "The name_scope might have been silently skipped inside tf.function."
    )

if __name__ == "__main__":
    test_name_scope_in_compiled_mode()
    print("Test passed: name_scope works correctly inside tf.function.")