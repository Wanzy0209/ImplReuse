import tensorflow as tf
import numpy as np

def test_all_v2_summary_ops_behavior():
    """
    Test case for tf.compat.v1.summary.all_v2_summary_ops.
    
    This test adapts the logic from the PyTorch MPS bug report (Issue 165257),
    which compared behavior across different states (CPU vs MPS).
    
    Here, we compare the behavior of all_v2_summary_ops across different
    execution contexts (Eager vs Graph), as the API implementation explicitly
    branches based on context.executing_eagerly().
    """
    
    # Setup: Create some V2 summary ops to populate the collection
    # This ensures the collection is not empty when we query it in graph mode.
    tf.compat.v1.summary.scalar("test_scalar", 1.0)
    tf.compat.v1.summary.histogram("test_hist", np.random.rand(10))

    # Define test scenarios mirroring the 'ops' list in the original bug report.
    # The original report iterated over random operations. Here we iterate over
    # execution contexts that trigger different code paths in the target API.
    contexts = [
        ("Eager Mode", lambda: None), 
        ("Graph Mode", lambda: tf.compat.v1.disable_eager_execution())
    ]

    print(f"{'Context':<20} {'Result Type':<20} {'Status'}")
    print("-" * 60)

    # Note: We must execute Eager mode first because disable_eager_execution 
    # is a global, irreversible change in TensorFlow 1.x compatibility.
    for name, setup_func in contexts:
        # Apply the context setup
        setup_func()
        
        # Call the API under test
        result = tf.compat.v1.summary.all_v2_summary_ops()
        
        # Verify behavior based on the context
        # The API returns a list in both Eager and Graph modes.
        # In Eager mode, the collection is typically empty (returns []).
        # In Graph mode, it returns the list of registered ops.
        if name == "Eager Mode":
            # Corrected expectation: API returns an empty list in Eager mode, not None
            is_correct = result == []
            result_str = "[]"
        else:
            is_correct = isinstance(result, list)
            result_str = f"list (len={len(result) if result else 0})"
            
        status = " OK" if is_correct else " BUG"
        print(f"{name:<20} {result_str:<20} {status}")
        
        # Assertions to act as the actual test verification
        if name == "Eager Mode":
            assert result == [], f"Expected empty list in {name}, got {result}"
        else:
            assert isinstance(result, list), f"Expected list in {name}, got {type(result)}"

if __name__ == "__main__":
    test_all_v2_summary_ops_behavior()