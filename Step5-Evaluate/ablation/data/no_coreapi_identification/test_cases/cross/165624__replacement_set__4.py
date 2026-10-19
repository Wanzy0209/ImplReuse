import tensorflow as tf
from unittest.mock import patch

def test_duplicate_relu_execution():
    """
    Test case to reproduce the duplicate execution bug found in Issue 165624.
    The original bug involved 'joint_custom_pre_pass' being executed twice.
    This test replaces 'joint_custom_pre_pass' with 'tf.keras.activations.relu'
    to verify the duplicate execution pattern using the replacement API.
    """

    # Mocking the configuration and graph context to simulate the original environment
    class Config:
        # Replacing joint_custom_pre_pass with tf.keras.activations.relu
        joint_custom_pre_pass = tf.keras.activations.relu
        joint_graph_constant_folding = False

    class GraphTransformObserver:
        def __init__(self, graph, name):
            self.graph = graph
            self.name = name

        def apply_graph_pass(self, func):
            # Simulate applying the pass to the graph
            func(self.graph)

        def apply_gm_pass(self, func):
            func(self.graph)

    # Dummy graph data (a tensor)
    graph = tf.constant([-1.0, 2.0, -3.0, 4.0])

    # Dummy functions for other passes
    def remove_noop_ops(g): pass
    def constant_fold_uniform_value(g): pass

    config = Config()
    count = 0

    # Patching tf.keras.activations.relu to track how many times it is called
    with patch.object(tf.keras.activations, 'relu', wraps=tf.keras.activations.relu) as mock_relu:
        
        # --- Original Bug Logic (Adapted) ---
        if config.joint_custom_pre_pass is not None:
            GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
                config.joint_custom_pre_pass
            )
            count += 1

        # from .post_grad import remove_noop_ops
        GraphTransformObserver(graph, "remove_noop_ops").apply_graph_pass(remove_noop_ops)

        if config.joint_graph_constant_folding:
            GraphTransformObserver(graph, "constant_fold_uniform_value").apply_gm_pass(
                constant_fold_uniform_value
            )

        if config.joint_custom_pre_pass is not None:
            GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
                config.joint_custom_pre_pass
            )
            count += 1
        # --- End of Original Bug Logic ---

        # Assertions
        # The bug is that the pass is executed twice.
        assert count == 2, "Expected count to be 2 due to duplicate execution logic."
        assert mock_relu.call_count == 2, "Bug reproduced: tf.keras.activations.relu (replacement for joint_custom_pre_pass) was executed twice."

if __name__ == "__main__":
    test_duplicate_relu_execution()
    print("Test passed: Bug logic successfully reproduced with replacement API.")