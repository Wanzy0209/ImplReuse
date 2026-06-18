import torch
import tensorflow as tf

# Setup variables from the original bug report
keys = range(10)
allowed = [0, 1, 2, 3]

def test_tpu_core_with_local_cell_conflict():
    """
    Test case for tf.compat.v1.tpu.core.
    
    This test preserves the logic from the PyTorch Dynamo bug report (Issue 161542),
    specifically the scoping issue where a list comprehension variable ('key') 
    is shadowed and accessed via 'nonlocal' in an inner function.
    
    We verify that tf.compat.v1.tpu.core functions correctly within this 
    complex scoping context, replacing the role of the graph_break or 
    compilation logic in the original issue.
    """
    
    def outer_fn(x):
        # Call the similar API: tf.compat.v1.tpu.core
        # In the original bug, torch._dynamo.graph_break() was called here.
        # We use the API to generate a device name based on input x.
        device_name = tf.compat.v1.tpu.core(x)
        
        # Reproduce the scoping logic that caused the bug:
        # 'key' is assigned from a list comprehension.
        # In Python 3, the iteration variable 'key' leaks into the local scope
        # of the comprehension, but the assignment to 'key' makes it a cell variable
        # for the inner function.
        key = [k for k in keys if k in allowed]

        def inner():
            nonlocal key
            # Use the similar API inside the inner function with the captured variable
            return tf.compat.v1.tpu.core(key[0])

        return device_name, inner()

    # Execute the function with a sample input
    # Original used torch.ones(3), here we use an integer for the TPU core ID
    result_outer, result_inner = outer_fn(5)

    # Assertions to verify correct behavior
    assert result_outer == "device:TPU_REPLICATED_CORE:5", \
        f"Expected 'device:TPU_REPLICATED_CORE:5', got '{result_outer}'"
    
    assert result_inner == "device:TPU_REPLICATED_CORE:0", \
        f"Expected 'device:TPU_REPLICATED_CORE:0', got '{result_inner}'"

if __name__ == "__main__":
    test_tpu_core_with_local_cell_conflict()
    print("Test passed successfully.")