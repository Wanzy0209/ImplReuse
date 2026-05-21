import torch
import tensorflow as tf
import numpy as np

def test_tf_enable_eager_execution():
    """
    Adapted from PyTorch Issue 167927:
    Original: torch.compile(fullgraph=True) should accept torch.compiler.disable
    
    This test verifies the behavior of tf.compat.v1.enable_eager_execution.
    In PyTorch, the issue is about the interaction between a compilation context 
    (torch.compile) and a context forcing eager execution (torch.compiler.disable).
    
    In TensorFlow, tf.compat.v1.enable_eager_execution is the global switch 
    that forces eager execution (similar to the 'disable' concept in PyTorch).
    We verify that this API successfully enables eager execution, allowing 
    operations to return concrete values immediately.
    """
    
    # Note: In TensorFlow 2.x, eager execution is enabled by default.
    # This API is primarily for compatibility or specific v1 scenarios.
    # We attempt to enable it to ensure the runtime is in the expected state.
    try:
        tf.compat.v1.enable_eager_execution()
    except RuntimeError as e:
        # If eager execution is already enabled or cannot be enabled due to 
        # prior graph usage, we catch the error to proceed with the verification.
        # "Eager execution cannot be enabled after TensorFlow APIs have been used..."
        print(f"Note: {e}")

    # Verify that eager execution is active.
    # This corresponds to the "torch.compiler.disable" state in PyTorch,
    # where graph compilation is skipped in favor of immediate execution.
    assert tf.executing_eagerly(), "Eager execution should be active."

    # Perform a simple operation to verify eager behavior.
    # In PyTorch, torch.compiler.disable forces the code to run immediately.
    # Here, enable_eager_execution forces the code to run immediately.
    x = tf.constant(6)
    y = tf.constant(7)
    z = tf.multiply(x, y)

    # In eager mode, we can access the value immediately via .numpy()
    # This mimics the behavior expected when disabling compilation in PyTorch.
    expected = 42
    assert z.numpy() == expected, f"Expected {expected}, got {z.numpy()}"

    print("Test passed: tf.compat.v1.enable_eager_execution successfully enables eager mode.")

if __name__ == "__main__":
    test_tf_enable_eager_execution()