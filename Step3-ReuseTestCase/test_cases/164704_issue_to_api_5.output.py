import torch
import tensorflow as tf

def test_tf_control_dependencies_scalar_ops():
    """
    This test case replicates the logic of the PyTorch bug report (Issue 164704)
    using TensorFlow. The original bug involves a divergence between eager and 
    compiled modes when handling scalar extraction and types (int vs float).
    
    Here, we leverage tf.control_dependencies to ensure the execution order 
    of the scalar operations within a tf.function (graph mode), mirroring the 
    intent of the original code's sentinel logic and the compiler's handling 
    of the operation chain.
    """
    
    # Enable eager execution explicitly (default in TF 2.x, but good for clarity)
    # The test focuses on tf.function behavior (graph mode).
    
    @tf.function
    def graph_mode_program(arg_0, arg_1):
        # Replicate: var_node_4 = torch.full((2, 3), 3, dtype=torch.int16)
        var_node_4 = tf.fill((2, 3), 3)
        var_node_4 = tf.cast(var_node_4, tf.int16)

        # Replicate: var_node_3 = torch.unique(var_node_4)
        # PyTorch unique handles N-D tensors, TF unique requires 1-D.
        # We reshape to flatten the input for unique.
        var_node_3, _ = tf.unique(tf.reshape(var_node_4, [-1]))

        # Replicate: var_node_2 = torch.squeeze(var_node_3)
        var_node_2 = tf.squeeze(var_node_3)

        # Replicate: var_node_7 = arg_0, var_node_8 = arg_1
        var_node_7 = arg_0
        var_node_8 = arg_1

        # Replicate: var_node_6 = torch.sub(var_node_7, var_node_8)
        var_node_6 = tf.subtract(var_node_7, var_node_8)

        # Replicate: var_node_10 = torch.full((1,), 3, dtype=torch.int16)
        var_node_10 = tf.fill((1,), 3)
        var_node_10 = tf.cast(var_node_10, tf.int16)

        # Replicate: var_node_9 = torch.squeeze(var_node_10)
        var_node_9 = tf.squeeze(var_node_10)

        # Replicate: var_node_5 = torch.add(var_node_6, var_node_9)
        var_node_5 = tf.add(var_node_6, var_node_9)

        # Replicate: var_node_1 = torch.div(var_node_2, var_node_5)
        # The PyTorch bug report indicates var_node_1 is int16.
        # We use floordiv to maintain integer type, similar to PyTorch's default integer division.
        var_node_1 = tf.math.floordiv(var_node_2, var_node_5)

        # Leverage the similar API: tf.control_dependencies
        # In the original bug, a sentinel was used to ensure gradient computation.
        # Here, we use control_dependencies to ensure the division (var_node_1) 
        # is computed before the final identity operation, enforcing the graph dependency.
        with tf.control_dependencies([var_node_1]):
            # This mimics the extraction/usage of the scalar value.
            # We use identity to carry the value forward within the dependency context.
            result = tf.identity(var_node_1)

        return result

    # Setup inputs matching the PyTorch logic (scalar int16)
    # PyTorch: torch.as_strided(torch.randint(5, 30, (1,)).to(torch.int16), (), ())
    # We use specific constants to ensure deterministic behavior for the test.
    arg_0 = tf.constant(20, dtype=tf.int16)
    arg_1 = tf.constant(5, dtype=tf.int16)

    # Calculate expected result manually based on the logic:
    # var_node_4 (full of 3s) -> unique -> 3
    # arg_0 (20) - arg_1 (5) = 15
    # var_node_10 (full of 3s) -> squeeze -> 3
    # 15 + 3 = 18
    # 3 // 18 = 0
    expected_result = tf.constant(0, dtype=tf.int16)

    # Execute in graph mode
    result_graph = graph_mode_program(arg_0, arg_1)

    # Assert the result matches the expected value
    # This verifies that the graph mode execution handled the types and dependencies correctly,
    # unlike the divergence in the original PyTorch bug.
    assert tf.equal(result_graph, expected_result).numpy(), \
        f"Graph mode failed: expected {expected_result.numpy()}, got {result_graph.numpy()}"

    print(" Test passed: Graph mode execution with control_dependencies succeeded.")

if __name__ == "__main__":
    test_tf_control_dependencies_scalar_ops()