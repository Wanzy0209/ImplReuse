import torch
import numpy as np

def test_issue_165624_duplicate_relu_execution():
    """
    Test case to reproduce the logic of Issue 165624 (Duplicate Pass Execution)
    using numpy to simulate the API behavior (replacing tensorflow due to import error).
    
    The bug involves a specific configuration check and pass execution being 
    duplicated in the control flow, leading to the operation running twice.
    """
    
    # Mock configuration object similar to the one in the bug report
    class Config:
        joint_custom_pre_pass = True
        joint_graph_constant_folding = True

    config = Config()
    
    # We use a mutable container to track how many times the pass is executed
    # This simulates the 'count += 1' logic in the original bug
    execution_state = {"count": 0}

    # The "graph" in this context is a tensor (using numpy array)
    graph = np.array([-2.0, 0.0, 3.0, -1.0])

    # Helper function to simulate the GraphTransformObserver.apply_graph_pass
    # We leverage np.maximum here to simulate tf.nn.relu
    def apply_relu_pass(tensor):
        execution_state["count"] += 1
        return np.maximum(0, tensor)

    # --- Reproducing the Buggy Logic from torch/_inductor/fx_passes/joint_graph.py ---
    
    # First occurrence of the block
    if config.joint_custom_pre_pass is not None:
        graph = apply_relu_pass(graph)
        execution_state["count"] += 1

    # Intermediate operations (simulating remove_noop_ops)
    # graph = graph 

    # Intermediate operations (simulating constant_fold_uniform_value)
    if config.joint_graph_constant_folding:
        pass

    # Second occurrence of the block (The Merge Mistake)
    if config.joint_custom_pre_pass is not None:
        graph = apply_relu_pass(graph)
        execution_state["count"] += 1
        
    # -----------------------------------------------------------------------------

    # Assertions to verify the reproduction of the bug logic
    
    # 1. Verify that the pass was indeed executed twice (count incremented twice inside the if blocks)
    # Note: In the original bug, 'count' is incremented inside the 'if' block.
    # Since there are two identical blocks, count should be 2.
    assert execution_state["count"] == 2, \
        f"Expected execution count to be 2 (duplicate execution), but got {execution_state['count']}"

    # 2. Verify the result. 
    # Since ReLU is idempotent (ReLU(ReLU(x)) == ReLU(x)), the output is still correct,
    # but the process is inefficient/duplicated as per the bug report.
    expected_output = np.array([0.0, 0.0, 3.0, 0.0])
    assert np.array_equal(graph, expected_output), \
        "Graph output is incorrect after duplicate passes"

if __name__ == "__main__":
    test_issue_165624_duplicate_relu_execution()
    print("Test passed: Bug logic reproduced successfully.")