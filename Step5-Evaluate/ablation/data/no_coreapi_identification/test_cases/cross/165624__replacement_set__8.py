import tensorflow as tf

# Mocking the GraphTransformObserver to make the code runnable in isolation
class GraphTransformObserver:
    def __init__(self, graph, name):
        self.graph = graph
        self.name = name

    def apply_graph_pass(self, func):
        # Simulate the pass execution.
        # tf.math.add expects two tensors, we pass the graph twice for the sake of the test.
        return func(self.graph, self.graph)

# Mocking other dependencies to preserve the structure of the original bug report
def remove_noop_ops(graph):
    return graph

def constant_fold_uniform_value(graph):
    return graph

# Mock config object
class Config:
    joint_graph_constant_folding = True

config = Config()

def test_merge_mistake_reproduction():
    """
    Test case to reproduce the logic of Issue 165624 where a merge mistake
    caused joint_custom_pre_pass to be executed twice.
    
    Here, joint_custom_pre_pass is replaced with tf.compat.v1.math.add.
    """
    graph = tf.constant([1.0, 2.0])
    count = 0

    # --- Start of Original Bug Logic (with replacements) ---
    
    # Replacement: joint_custom_pre_pass -> tf.compat.v1.math.add
    if tf.compat.v1.math.add is not None:
        GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
            tf.compat.v1.math.add
        )
        count += 1

    # from .post_grad import remove_noop_ops
    GraphTransformObserver(graph, "remove_noop_ops").apply_graph_pass(remove_noop_ops)

    if config.joint_graph_constant_folding:
        GraphTransformObserver(graph, "constant_fold_uniform_value").apply_graph_pass(
            constant_fold_uniform_value
        )

    # Replacement: joint_custom_pre_pass -> tf.compat.v1.math.add
    # This is the duplicated block causing the bug.
    if tf.compat.v1.math.add is not None:
        GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
            tf.compat.v1.math.add
        )
        count += 1
        
    # --- End of Original Bug Logic ---

    # The bug is that the operation is executed twice.
    # We assert count == 2 to confirm the reproduction of the bug logic.
    assert count == 2, f"Expected 2 executions (bug reproduction), got {count}"
    print("Test passed: Bug logic reproduced (double execution of tf.math.add).")

if __name__ == "__main__":
    test_merge_mistake_reproduction()