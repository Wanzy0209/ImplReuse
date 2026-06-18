import torch
import tensorflow as tf

def test_executing_eagerly_temporary_context():
    """
    Test case adapted from PyTorch MemPool bug (Issue 167745).
    
    Original Bug Logic:
    The bug highlights failures when MemPool objects are created temporarily 
    within a 'with' statement (e.g., `with use_mem_pool(MemPool(...))`).
    It also tests nested contexts to ensure correct state propagation.

    Adaptation for tf.compat.v1.executing_eagerly_outside_functions:
    This test verifies that the API correctly identifies the outer eager context
    even when invoked from within a temporarily created or nested graph context
    (tf.function), mirroring the scope management issues in the original bug.
    """
    
    # Ensure we are in eager mode (default in TF2)
    assert tf.executing_eagerly(), "Test must be run in eager mode"

    # Test 1: Mimic the temporary object pattern
    # PyTorch: with torch.cuda.use_mem_pool(torch.cuda.MemPool(pool1)):
    # TensorFlow: Create a temporary tf.function (inline) and call it immediately.
    print("Test 1: Temporary function context")
    
    # The lambda represents the temporary object construction logic
    # We wrap the check in a tf.function to enter the "inner" context (Graph mode)
    temp_func = tf.function(lambda: tf.compat.v1.executing_eagerly_outside_functions())
    
    # Execute the temporary context
    result = temp_func()
    
    # The function should return True because the outer context is eager
    assert bool(result.numpy()), "Outer context should be eager even inside temporary function"
    print("Test 1 passed")

    # Test 2: Mimic nested contexts
    # PyTorch: x1 (Outer) -> with pool1 -> with pool2
    # TensorFlow: Check (Outer) -> tf.function (Inner) -> Check
    
    print("Test 2: Nested context check")
    
    # Outer state check (equivalent to x1 outside pools)
    outer_check = tf.compat.v1.executing_eagerly_outside_functions()
    assert bool(outer_check.numpy()), "Outer context should be eager"

    # Inner state check (equivalent to being inside Pool 1/Pool 2)
    @tf.function
    def nested_check():
        # Inside the "inner pool" (Graph mode)
        # We verify that the API still sees the outer eager context
        return tf.compat.v1.executing_eagerly_outside_functions()

    inner_result = nested_check()
    assert bool(inner_result.numpy()), "Outer context should still be eager from inside nested function"
    print("Test 2 passed")

if __name__ == "__main__":
    test_executing_eagerly_temporary_context()