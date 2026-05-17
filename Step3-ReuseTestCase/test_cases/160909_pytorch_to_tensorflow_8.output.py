import tensorflow as tf

def test_name_scope_with_compilation():
    """
    Adapted from PyTorch Issue 160909.
    
    Original Logic:
    1. Define a custom backend logic.
    2. Compile a model moved to a custom device (PrivateUse1).
    3. Run the model.
    4. Bug: Compiler uses 'meta' tensors (for tracing) but fails dispatch 
       against the custom device logic.

    Adapted Logic for tf.keras.name_scope:
    1. Define a model logic using tf.function (TF's compilation mechanism).
    2. Wrap the execution in tf.keras.name_scope (The target API).
    3. Run the model.
    4. Verify that the scope context is respected during the tracing/execution phase,
       ensuring no context leakage or mismatch similar to the device mismatch in PyTorch.
    """

    # Define a simple model that performs an operation similar to the bug report (repeat_interleave)
    @tf.function
    def my_model(x):
        # In PyTorch, the error occurred in repeat_interleave -> flatten
        # We use tf.repeat which is the semantic equivalent
        return tf.repeat(x, repeats=2, axis=1)

    # Input data mimicking the shape in the bug report: (1, 8, 3, 128)
    # Note: We use standard TF tensors here as there is no direct "PrivateUse1" in TF
    # without registering a custom device, which is out of scope for a simple API test.
    data = tf.ones((1, 8, 3, 128))

    # The target API: tf.keras.name_scope
    # This acts as the context wrapper, similar to how the custom backend/device context
    # wrapped the execution in PyTorch.
    with tf.keras.name_scope("custom_context_scope"):
        result = my_model(data)

    # Assertions to verify correct behavior
    # 1. Check output shape (repeat along axis 1 doubles the dimension)
    assert result.shape == (1, 16, 3, 128), f"Expected shape (1, 16, 3, 128), got {result.shape}"

    # 2. Verify that the operations inside the compiled function respect the scope
    # We check the concrete function graph to ensure the scope was applied correctly
    # during the tracing phase (analogous to the 'meta' phase in PyTorch).
    concrete_func = my_model.get_concrete_function(data)
    graph = concrete_func.graph
    
    # Look for the Repeat operation in the graph
    found_repeat_in_scope = False
    for op in graph.get_operations():
        if 'Repeat' in op.type:
            # Check if the operation name includes the scope we defined
            # This verifies the context was not lost during compilation/tracing
            if 'custom_context_scope' in op.name:
                found_repeat_in_scope = True
                break
    
    assert found_repeat_in_scope, "Repeat operation was not found within the expected name scope. Context might have been lost during tracing."

    print("Test passed: tf.keras.name_scope context respected during tf.function compilation.")

if __name__ == "__main__":
    test_name_scope_with_compilation()