import torch
import tensorflow as tf

def test_name_scope_in_compilation():
    """
    Adapts the logic of the PyTorch bug report to TensorFlow.
    
    In PyTorch, using `torch.compiler.disable` inside a function compiled with 
    `torch.compile(fullgraph=True)` raises an error because `fullgraph=True` 
    forbids graph breaks, while `disable` forces one.
    
    This test verifies that the similar TensorFlow API, `tf.keras.name_scope`, 
    works correctly inside a compiled function (tf.function), ensuring that 
    the context manager is compatible with the compilation process.
    """
    
    # Define a function that uses the similar API (tf.keras.name_scope)
    # This mimics the usage of torch.compiler.disable in the original issue.
    def inner_function(x):
        with tf.keras.name_scope("custom_scope"):
            return x + 1

    # Compile the function using tf.function (equivalent to torch.compile)
    # We use jit_compile=True to mimic the strictness of fullgraph=True
    compiled_function = tf.function(inner_function, jit_compile=True)

    # Execute the compiled function
    input_tensor = tf.constant(5.0)
    result = compiled_function(input_tensor)

    # Verify the result is correct and no error was raised
    # The PyTorch bug would raise torch._dynamo.exc.Unsupported here.
    assert result.numpy() == 6.0, "tf.keras.name_scope should work inside tf.function"

if __name__ == "__main__":
    test_name_scope_in_compilation()
    print("Test passed: tf.keras.name_scope is compatible with tf.function.")