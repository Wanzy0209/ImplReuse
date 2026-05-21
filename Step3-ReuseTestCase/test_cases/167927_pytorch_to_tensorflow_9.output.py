import torch
import tensorflow as tf

def test_name_scope_in_compilation():
    """
    Adapts the logic of the PyTorch bug report:
    Using a context manager (torch.compiler.disable -> tf.name_scope)
    inside a strict compilation context (torch.compile(fullgraph=True) -> tf.function).
    
    In PyTorch, this raised an error. In TensorFlow, we verify that 
    tf.name_scope works correctly within the graph context.
    """
    
    # Define a function that uses the similar API (tf.name_scope)
    # This corresponds to using torch.compiler.disable in the original issue
    def inner_func(x):
        with tf.name_scope("custom_scope"):
            return x + 1

    # Apply the compilation equivalent (tf.function)
    # This corresponds to torch.compile(fullgraph=True)
    compiled_func = tf.function(inner_func)

    # Execute the function
    input_tensor = tf.constant(1.0)
    result = compiled_func(input_tensor)

    # 1. Verify the function executes successfully (no crash like the PyTorch bug)
    assert result.numpy() == 2.0, "Function execution failed"

    # 2. Verify the context manager behavior (scope naming) is preserved in the graph
    # This ensures the API did its job inside the compiled context
    concrete_func = compiled_func.get_concrete_function(input_tensor)
    graph = concrete_func.graph
    ops_in_graph = [op.name for op in graph.get_operations()]
    
    # Check if any operation in the graph carries the scope name
    has_scope = any("custom_scope" in op_name for op_name in ops_in_graph)
    assert has_scope, f"tf.name_scope was not applied. Graph ops: {ops_in_graph}"

    print("Test passed: tf.name_scope works correctly inside tf.function.")

if __name__ == "__main__":
    test_name_scope_in_compilation()