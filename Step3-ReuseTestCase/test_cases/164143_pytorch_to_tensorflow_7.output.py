import torch
import tensorflow as tf

def test_name_scope_with_tf_function():
    """
    Adapts the PyTorch bug report (DebugMode silently disables torch.compile)
    to TensorFlow.

    In PyTorch, a dispatch mode (DebugMode) caused compilation to be skipped.
    In TensorFlow, name_scope has a 'skip_on_eager' mode. 
    This test verifies that name_scope does NOT silently skip when used 
    inside tf.function (graph mode), ensuring the mode logic is correct.
    """
    
    @tf.function
    def compiled_func(x):
        # With skip_on_eager=True (default), this scope is skipped in eager mode.
        # We verify it is NOT skipped inside tf.function.
        with tf.keras.backend.name_scope("debug_scope"):
            return x + 1

    # Run the compiled function
    result = compiled_func(1.0)
    
    # Inspect the graph to ensure the scope was applied
    concrete_func = compiled_func.get_concrete_function(tf.TensorSpec(shape=[], dtype=tf.float32))
    graph_def = concrete_func.graph.as_graph_def()
    
    # Assert that the scope name appears in the graph nodes
    has_scope = any("debug_scope" in node.name for node in graph_def.node)
    
    # If this assertion fails, it means name_scope was silently skipped
    # inside the compiled context, similar to the PyTorch bug.
    assert has_scope, "name_scope was silently skipped inside tf.function"

if __name__ == "__main__":
    test_name_scope_with_tf_function()
    print("Test passed: name_scope works correctly with tf.function.")