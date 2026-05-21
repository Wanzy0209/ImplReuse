import torch
import tensorflow as tf

def test_name_scope_functionality():
    """
    Adapts the logic of wrapping a computation block (originally torch.compile)
    to the TensorFlow name_scope API.
    
    Original Bug Context: Performance regression in torch.compile for mxfp8 quantization.
    Adaptation: Verify that tf.keras.backend.name_scope correctly wraps operations
    and applies the naming hierarchy, which is its core responsibility.
    """
    
    # Define a mode/scope name similar to the benchmark argument in the bug report
    mode_name = "dim0_mxfp8_floor"
    
    # Setup input data (mimicking the tensor operations in the benchmark)
    # Original: M 16384 K 16384 BLOCK_SIZE 32
    # Adaptation: Smaller tensor for functional unit testing
    input_tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
    
    # Apply the API under test: tf.keras.backend.name_scope
    # This acts as the context manager for the operation, similar to how 
    # torch.compile wraps the execution context.
    with tf.keras.backend.name_scope(mode_name):
        # Perform a computation (mimicking the quantization/cast logic)
        # We use a simple square operation to generate a graph node.
        result = tf.square(input_tensor)
        
        # Verify that the operation falls within the scope
        # The name of the resulting tensor should be prefixed by the scope name
        assert result.name.startswith(f"{mode_name}/"), \
            f"Expected operation name to start with '{mode_name}/', but got '{result.name}'"

    print(f"Test Passed: Operation '{result.name}' correctly scoped under '{mode_name}'")

if __name__ == "__main__":
    test_name_scope_functionality()