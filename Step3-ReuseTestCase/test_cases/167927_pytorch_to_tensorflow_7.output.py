import torch
import tensorflow as tf

def test_name_scope_in_strict_compile():
    """
    Test case adapted from PyTorch issue #167927.
    
    Original Issue: torch.compile(fullgraph=True) should accept torch.compiler.disable.
    The bug was that using a disabling context manager inside a strict compilation
    mode caused an error.
    
    Adapted Logic: Verify that the similar API (tf.keras.backend.name_scope) 
    works correctly inside a strict compilation context (tf.function with jit_compile=True).
    This ensures the context manager integrates with the graph without causing 
    unsupported errors or unexpected graph breaks.
    """
    
    # Define a function with strict compilation (equivalent to fullgraph=True)
    @tf.function(jit_compile=True)
    def func(x):
        # Use the similar API: tf.keras.backend.name_scope
        # In the PyTorch bug, torch.compiler.disable is used here.
        # We verify that this context manager is accepted by the strict compiler.
        with tf.keras.backend.name_scope("adapted_scope"):
            return x + 1

    input_val = tf.constant(5.0)
    
    # Execute the function
    # In the PyTorch bug, this raised torch._dynamo.exc.Unsupported
    try:
        result = func(input_val)
    except Exception as e:
        raise AssertionError(f"tf.keras.backend.name_scope failed inside strict compilation: {e}")

    # Assertion 1: Functional correctness
    assert result.numpy() == 6.0, "Function output is incorrect"
    
    # Assertion 2: Verify the scope was actually applied to the graph
    # This confirms the context manager was not skipped or treated as an error.
    concrete_func = func.get_concrete_function(input_val)
    graph = concrete_func.graph
    
    has_scope = any("adapted_scope" in op.name for op in graph.get_operations())
    assert has_scope, "tf.keras.backend.name_scope did not affect the graph structure."

    print("Test passed: tf.keras.backend.name_scope is compatible with strict compilation (jit_compile=True).")

if __name__ == "__main__":
    test_name_scope_in_strict_compile()