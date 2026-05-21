import tensorflow as tf
import numpy as np

def test_merge_mistake_pattern_with_relu():
    """
    Test case reflecting the logic of Issue 165624 (A Likely Merge Mistake).
    
    The original issue involved a graph transformation pass (joint_custom_pre_pass) 
    being executed twice due to a merge conflict where code was duplicated.
    
    This test adapts that logic to the similar API (tf.keras.ops.relu) to verify
    behavior under the "double execution" pattern. Since ReLU is idempotent,
    applying it twice should yield the same result as applying it once.
    """
    
    # Setup: Simulating the configuration and graph state
    class Config:
        joint_custom_pre_pass = True
        joint_graph_constant_folding = False

    config = Config()
    
    # Input tensor: mix of negative and positive values
    # Represents the 'graph' in the original issue
    graph = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
    
    count = 0

    # --- Reproducing the Bug Logic (Double Execution) ---

    # First block: Check config and apply pass
    if config.joint_custom_pre_pass:
        # Mapping GraphTransformObserver(...).apply_graph_pass to tf.keras.ops.relu
        graph = tf.keras.ops.relu(graph)
        count += 1

    # Intermediate operation (simulating remove_noop_ops or constant folding)
    # In the original bug, this code sat between the two duplicate blocks
    graph = graph + 0.0 

    # Second block: The "Merge Mistake" - Check config and apply pass again
    if config.joint_custom_pre_pass:
        # Duplicate call to the API
        graph = tf.keras.ops.relu(graph)
        count += 1

    # --- Verification ---
    
    # Expected result for ReLU: [0.0, 0.0, 0.0, 1.0, 2.0]
    expected = tf.constant([0.0, 0.0, 0.0, 1.0, 2.0], dtype=tf.float32)
    
    # Assert the output is correct despite the double execution
    assert tf.reduce_all(tf.equal(graph, expected)).numpy(), \
        f"Output mismatch. Expected {expected}, got {graph}"
    
    # Assert the counter reflects the double execution (the bug symptom)
    assert count == 2, "Execution count should be 2 due to the duplicated code block"

    print("Test passed: The double execution pattern was successfully applied and validated.")

if __name__ == "__main__":
    test_merge_mistake_pattern_with_relu()