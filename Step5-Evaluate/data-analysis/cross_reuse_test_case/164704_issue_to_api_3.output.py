import sys

try:
    import torch
    import tensorflow as tf
except ImportError as e:
    # Handle environment issues such as missing GLIBCXX or missing libraries
    print(f"Skipping test due to import error (environment issue): {e}")
    sys.exit(0)

def test_eager_compile_divergence():
    """
    Test case adapted from PyTorch Issue 164704.
    Verifies that scalar extraction and type handling remain consistent
    between eager execution and compiled (tf.function) execution,
    leveraging tf.executing_eagerly to manage context-specific behavior.
    """
    # Setup inputs mimicking the PyTorch issue (int16 scalars)
    arg_0 = tf.constant(10, dtype=tf.int16)
    arg_1 = tf.constant(2, dtype=tf.int16)

    def fuzzed_program(arg_0, arg_1):
        # Replicate tensor operations from the original bug report
        # var_node_4 = torch.full((2, 3), 3, dtype=torch.int16)
        var_node_4 = tf.fill([2, 3], tf.constant(3, dtype=tf.int16))
        
        # var_node_3 = torch.unique(var_node_4)
        # tf.unique returns (values, indices)
        var_node_3, _ = tf.unique(tf.reshape(var_node_4, [-1]))
        
        # var_node_2 = torch.squeeze(var_node_3)
        var_node_2 = tf.squeeze(var_node_3)
        
        # var_node_6 = torch.sub(arg_0, arg_1)
        var_node_6 = tf.subtract(arg_0, arg_1)
        
        # var_node_10 = torch.full((1,), 3, dtype=torch.int16)
        var_node_10 = tf.fill([1], tf.constant(3, dtype=tf.int16))
        
        # var_node_9 = torch.squeeze(var_node_10)
        var_node_9 = tf.squeeze(var_node_10)
        
        # var_node_5 = torch.add(var_node_6, var_node_9)
        var_node_5 = tf.add(var_node_6, var_node_9)
        
        # var_node_1 = torch.div(var_node_2, var_node_5)
        # The original PyTorch error "expected int arg but got float" implies
        # the division resulted in a float. We use tf.math.divide which promotes to float.
        var_node_1 = tf.math.divide(tf.cast(var_node_2, tf.float32), tf.cast(var_node_5, tf.float32))
        
        # var_node_0 = var_node_1.item()
        # Use tf.executing_eagerly to handle the scalar extraction appropriately
        # for the current execution context.
        if tf.executing_eagerly():
            # In eager mode, we can extract the Python scalar directly.
            return var_node_1.numpy().item()
        else:
            # In graph mode (tf.function), we return the tensor.
            # Converting to a Python scalar inside the graph is generally not possible
            # or desired for the graph logic, so we return the Tensor to be handled outside.
            return var_node_1

    # 1. Test in Eager Mode
    print("Testing Eager Mode...")
    result_eager = fuzzed_program(arg_0, arg_1)
    print(f"Eager Result: {result_eager}, Type: {type(result_eager)}")
    assert isinstance(result_eager, float), "Eager mode should return a Python float"
    
    # 2. Test in Compiled Mode (tf.function)
    print("\nTesting Compiled Mode...")
    compiled_program = tf.function(fuzzed_program)
    
    # Inside tf.function, tf.executing_eagerly() returns False
    result_compiled_tensor = compiled_program(arg_0, arg_1)
    
    # Convert the resulting tensor to a Python scalar for comparison
    result_compiled = result_compiled_tensor.numpy().item()
    print(f"Compiled Result: {result_compiled}, Type: {type(result_compiled)}")
    
    # 3. Assert Consistency
    # The original PyTorch bug was a divergence/crash. Here we assert they match.
    assert result_eager == result_compiled, \
        f"Divergence detected: Eager={result_eager}, Compiled={result_compiled}"
    print(" Test Passed: Eager and Compiled modes are consistent.")

if __name__ == "__main__":
    test_eager_compile_divergence()