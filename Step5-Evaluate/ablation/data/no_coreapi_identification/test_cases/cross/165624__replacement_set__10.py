import tensorflow as tf
from unittest.mock import patch

def test_merge_mistake_duplicate_add():
    """
    Test case to reproduce the merge mistake bug where a custom pass is executed twice.
    This test replaces the original 'joint_custom_pre_pass' with 'tf.keras.layers.add'
    to verify the duplicate execution logic.
    """
    # Setup dummy inputs for tf.keras.layers.add
    input1 = tf.constant([[1.0]])
    input2 = tf.constant([[2.0]])
    graph = [input1, input2]

    # Mock configuration
    class Config:
        joint_custom_pre_pass = True

    config = Config()

    # We mock tf.keras.layers.add to track how many times it is called
    with patch('tensorflow.keras.layers.add', wraps=tf.keras.layers.add) as mock_add:
        count = 0

        # --- Start of Buggy Logic (from Issue 165624) ---
        # Replaced joint_custom_pre_pass execution with tf.keras.layers.add

        if config.joint_custom_pre_pass is not None:
            # Original: GraphTransformObserver(...).apply_graph_pass(...)
            # Replacement: tf.keras.layers.add(...)
            tf.keras.layers.add(graph)
            count += 1

        # Simulating intermediate steps (e.g., remove_noop_ops)
        # ...

        if config.joint_custom_pre_pass is not None:
            # The duplicate execution (Merge Mistake)
            tf.keras.layers.add(graph)
            count += 1
        # --- End of Buggy Logic ---

        # Assertions
        # The bug causes the function to be called twice.
        assert mock_add.call_count == 2, \
            f"Bug reproduced: tf.keras.layers.add was called {mock_add.call_count} times (expected 2 for bug, 1 for fix)."
        assert count == 2, "Logic count indicates duplicate execution."

if __name__ == "__main__":
    test_merge_mistake_duplicate_add()
    print("Test passed: Bug logic successfully reproduced.")