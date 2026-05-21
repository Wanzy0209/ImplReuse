import torch
import tensorflow as tf
import numpy as np

def test_tf_name_scope_with_clamp():
    """
    Adapted test case for tf.keras.name_scope based on the PyTorch Inductor bug.
    
    Original Bug: torch.compile failed with NameError when using .item() on a float tensor arg.
    Adaptation: Uses tf.function (TF's compilation) and tf.keras.name_scope (the requested similar API)
    to verify that the operation executes correctly without code generation errors.
    """
    
    # Define the function logic
    # In TensorFlow, tf.function is the equivalent of torch.compile for graph execution.
    # We use tf.keras.name_scope as the context manager similar to the requested API.
    @tf.function
    def f(x, max_val):
        with tf.keras.name_scope("clamp_operation"):
            # In PyTorch: y = torch.clamp(x, 0, max_val.item())
            # In TF, we use tf.clip_by_value. 
            # Note: Directly using .item() (or .numpy()) inside a tf.function graph 
            # is generally not supported for control flow unless using tf.py_function.
            # We pass the tensor max_val directly, which is the idiomatic TF equivalent 
            # for passing a scalar tensor argument to an operation.
            y = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=max_val)
        return y

    # Setup inputs
    # Using CPU to ensure the test runs everywhere without CUDA requirements
    x = tf.random.normal((10, 20, 30))
    max_val = tf.constant(5.0)

    # Execute the compiled function
    # This verifies that the graph generation (similar to Triton kernel generation)
    # handles the arguments correctly without NameErrors.
    result = f(x, max_val)

    # Assertions to verify correctness
    assert result.shape == (10, 20, 30), "Output shape mismatch"
    
    # Verify clamping logic
    # All values should be <= 5.0
    assert tf.reduce_all(result <= 5.0).numpy(), "Values exceed max_val"
    # All values should be >= 0.0
    assert tf.reduce_all(result >= 0.0).numpy(), "Values are below min_val"

    print("Test passed: tf.keras.name_scope and tf.function handled the operation correctly.")

if __name__ == "__main__":
    test_tf_name_scope_with_clamp()