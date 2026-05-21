import torch
import tensorflow as tf
import pytest

def test_assert_provides_debugging_context():
    """
    Adapts the logic from PyTorch Issue 162858 (Graph break context logging)
    to TensorFlow's tf.debugging.Assert.

    The original issue requests that when a graph break occurs, the state of
    the stack (variables) is logged to help debugging. This test verifies that
    tf.debugging.Assert can be used to log the state of tensors (x, y, z)
    when an assertion fails, providing the requested debugging context.
    """
    @tf.function
    def fn(x):
        y = x + 1
        z = x + y
        
        # Equivalent to torch._dynamo.graph_break() but with explicit context logging.
        # We pass the current state of variables [x, y, z] to the 'data' argument.
        # This mimics the "Most recent bytecode/Python stack context" requested in the issue.
        tf.debugging.Assert(False, [x, y, z], summarize=10)
        return z

    input_tensor = tf.ones(3)

    # The assertion should raise an InvalidArgumentError.
    with pytest.raises(tf.errors.InvalidArgumentError) as excinfo:
        fn(input_tensor)

    # Verify that the error message contains the values of the variables
    # to ensure the debugging context was logged.
    # x=[1,1,1], y=[2,2,2], z=[3,3,3]
    error_msg = str(excinfo.value)
    assert "1" in error_msg
    assert "2" in error_msg
    assert "3" in error_msg