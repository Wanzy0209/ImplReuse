import torch
import tensorflow as tf

def test_merge_mistake_duplicate_relu():
    """
    Reproduces the logic of Issue 165624 (Duplicate execution due to merge mistake)
    using tf.keras.activations.relu as the operation being duplicated.
    
    The bug in torch/_inductor/fx_passes/joint_graph.py resulted in 
    `joint_custom_pre_pass` being executed twice due to a merge conflict.
    
    This test simulates that structural bug using `tf.keras.activations.relu`
    to verify that such a logic error would produce incorrect results.
    """
    
    # Simulating the 'config' flag from the original issue
    config_apply_relu = True

    # Input data
    # We use a value that becomes negative after the intermediate step
    # to make the double application of ReLU detectable.
    # Input: 10.0
    graph_input = tf.constant([10.0])

    # --- Buggy Code Pattern (from Issue 165624) ---
    # The original bug had the 'if config.joint_custom_pre_pass' block twice.
    # Here, we apply 'relu' twice.
    def buggy_pass(graph, config):
        count = 0
        
        # First block (Original location)
        if config:
            graph = tf.keras.activations.relu(graph)
            count += 1

        # Intermediate operation (e.g., remove_noop_ops or constant folding)
        # Subtracting 15 ensures the result goes negative, making the 2nd ReLU significant
        graph = graph - 15.0

        # Second block (Mistakenly added during merge, old one not removed)
        if config:
            graph = tf.keras.activations.relu(graph) # Duplicate!
            count += 1
            
        return graph, count

    # --- Fixed Code Pattern ---
    def fixed_pass(graph, config):
        count = 0
        
        if config:
            graph = tf.keras.activations.relu(graph)
            count += 1

        graph = graph - 15.0

        return graph, count

    # Run buggy version
    result_buggy, count_buggy = buggy_pass(graph_input, config_apply_relu)
    
    # Run fixed version
    result_fixed, count_fixed = fixed_pass(graph_input, config_apply_relu)

    # Assertions
    
    # 1. Verify execution count (The bug report mentions 'count += 1' twice)
    assert count_buggy == 2, "Buggy version should execute the pass twice"
    assert count_fixed == 1, "Fixed version should execute the pass once"

    # 2. Verify tensor values
    # Fixed logic: relu(10.0) - 15.0 = 10.0 - 15.0 = -5.0
    # Buggy logic: relu(relu(10.0) - 15.0) = relu(-5.0) = 0.0
    expected_buggy_val = 0.0
    expected_fixed_val = -5.0
    
    assert result_buggy.numpy()[0] == expected_buggy_val, \
        f"Buggy version result should be {expected_buggy_val}, got {result_buggy.numpy()[0]}"
        
    assert result_fixed.numpy()[0] == expected_fixed_val, \
        f"Fixed version result should be {expected_fixed_val}, got {result_fixed.numpy()[0]}"

    print("Test Passed: Successfully detected the merge mistake pattern using tf.keras.activations.relu.")

if __name__ == "__main__":
    test_merge_mistake_duplicate_relu()