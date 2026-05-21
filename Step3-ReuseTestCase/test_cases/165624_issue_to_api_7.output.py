import tensorflow as tf
from unittest.mock import patch

def test_merge_mistake_duplication_with_tf_add():
    """
    Reproduces the logic of Issue 165624 (Merge Mistake) using tf.compat.v1.math.add.
    The bug involves a conditional block being duplicated, causing an operation
    to execute twice.
    """
    
    # We mock the underlying implementation to track execution count.
    # This simulates checking if the API is called redundantly.
    with patch('tensorflow.python.ops.math_ops.add') as mock_add:
        # Setup mock return value
        mock_add.return_value = tf.constant(10.0)
        
        # --- Start of Buggy Logic Pattern (from Issue 165624) ---
        
        # Configuration flag (equivalent to config.joint_custom_pre_pass is not None)
        config_flag = True 
        count = 0
        result = tf.constant(0.0)
        
        # First conditional block
        if config_flag:
            # Using tf.compat.v1.math.add as the operation being duplicated
            # Equivalent to: GraphTransformObserver(...).apply_graph_pass(...)
            result = tf.compat.v1.math.add(tf.constant(5.0), tf.constant(5.0))
            count += 1
        
        # Intermediate operations (e.g., remove_noop_ops)
        # In this test, we just pass the result through
        _ = result 
        
        # Second conditional block (The Merge Mistake)
        if config_flag:
            # Duplicate call to tf.compat.v1.math.add
            result = tf.compat.v1.math.add(tf.constant(5.0), tf.constant(5.0))
            count += 1
        
        # --- End of Buggy Logic Pattern ---

        # Assertions to verify the bug reproduction
        # 1. The operation was called twice
        assert mock_add.call_count == 2, \
            f"Bug Reproduction Failed: Expected 2 calls, got {mock_add.call_count}"
        
        # 2. The counter reflects the double execution
        assert count == 2, \
            f"Bug Reproduction Failed: Expected count 2, got {count}"
        
        print("Test Case Passed: Successfully reproduced the merge mistake logic using tf.compat.v1.math.add")

if __name__ == "__main__":
    test_merge_mistake_duplication_with_tf_add()