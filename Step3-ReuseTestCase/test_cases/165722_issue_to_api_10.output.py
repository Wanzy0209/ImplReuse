import torch
import tensorflow as tf
import pytest

def test_operator_not_allowed_in_graph_robustness():
    """
    Test that OperatorNotAllowedInGraphError is raised consistently
    even after graph resets, ensuring no dangling pointer issues
    similar to the PyTorch OBJECT_ALIASING bug (Issue #165722).

    The original bug involved a guard storing a PyObject* without
    reference counting, leading to a crash after torch._dynamo.reset().
    This test verifies that TensorFlow's error handling for graph
    violations remains stable after clearing/resetting graph state.
    """
    # Define a function that triggers OperatorNotAllowedInGraphError
    # by using a tensor as a Python boolean (illegal in Graph mode).
    @tf.function
    def illegal_bool_check(t):
        if t:  # This line raises the error in Graph mode
            return 1
        return 0

    tensor = tf.constant([1, 2, 3])

    # Step 1: Initial call should raise OperatorNotAllowedInGraphError
    with pytest.raises(tf.errors.OperatorNotAllowedInGraphError):
        illegal_bool_check(tensor)

    # Step 2: Simulate a reset of the compiler/graph state.
    # In PyTorch, this is torch._dynamo.reset().
    # In TensorFlow, we reset the default graph to clear cached state.
    tf.compat.v1.reset_default_graph()

    # Re-define the function and tensor after reset to ensure clean state
    @tf.function
    def illegal_bool_check_after_reset(t):
        if t:
            return 1
        return 0

    tensor_after_reset = tf.constant([1, 2, 3])

    # Step 3: Call again after reset.
    # The PyTorch bug would cause a segfault here due to a dangling pointer.
    # We assert that TF correctly raises the error again, proving stability.
    with pytest.raises(tf.errors.OperatorNotAllowedInGraphError):
        illegal_bool_check_after_reset(tensor_after_reset)

if __name__ == "__main__":
    test_operator_not_allowed_in_graph_robustness()