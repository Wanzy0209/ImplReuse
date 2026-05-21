import torch
import tensorflow as tf
import gc

def test_summary_value_reference_safety():
    """
    Test case to verify that tf.compat.v1.Summary.Value handles object 
    lifecycles correctly, analogous to the fix for the PyTorch 
    OBJECT_ALIASING dangling pointer bug (Issue #165722).
    
    The PyTorch bug occurred when a guard stored a raw PyObject* and 
    the environment was reset, leaving a dangling pointer.
    
    This test verifies that tf.compat.v1.Summary.Value, which stores 
    a TensorProto, maintains data integrity even after the TensorFlow 
    graph is reset.
    """
    # Ensure we are in a state where graph reset is possible
    tf.compat.v1.disable_v2_behavior()

    # 1. Create a context (Graph) and a Tensor
    with tf.compat.v1.Graph().as_default():
        # Create a tensor
        tensor = tf.constant([1.0, 2.0, 3.0], name="test_tensor")
        
        # Convert to TensorProto (required by Summary.Value)
        # This mimics the "value" being captured by the guard
        tensor_proto = tf.make_tensor_proto(tensor)
        
        # 2. Instantiate the Similar API: tf.compat.v1.Summary.Value
        # This object stores the tensor data.
        summary_value = tf.compat.v1.Summary.Value(
            tag="dummy_tag",
            tensor=tensor_proto
        )
        
        # Store the value in a variable that persists outside the graph context
        # This simulates the 'Guard' holding the reference
        stored_value = summary_value

    # 3. Perform the "Reset" operation
    # Analogous to torch._dynamo.reset() in the bug report.
    # This clears the default graph, potentially invalidating raw pointers 
    # to tensors within that graph if they were not reference-counted properly.
    tf.compat.v1.reset_default_graph()
    
    # Force garbage collection to expose potential dangling pointers
    gc.collect()

    # 4. Verify the stored reference is still valid
    # In the PyTorch bug, accessing _first_tensor here would cause a crash 
    # or undefined behavior because it was a raw pointer to a freed object.
    # We expect tf.compat.v1.Summary.Value to be safe because it owns 
    # the TensorProto data (copy/ownership semantics).
    try:
        assert stored_value is not None, "Summary.Value object was garbage collected unexpectedly"
        assert stored_value.tag == "dummy_tag", "Summary.Value metadata was corrupted"
        
        # Verify the actual data payload is intact
        # Accessing float_val checks the memory validity of the stored proto
        assert len(stored_value.tensor.float_val) == 3, "Tensor data length mismatch"
        assert stored_value.tensor.float_val[0] == 1.0, "Tensor data corruption detected"
        
        print("Test Passed: tf.compat.v1.Summary.Value maintained data integrity after graph reset.")
        
    except Exception as e:
        print(f"Test Failed: Reference became invalid or data corrupted - {e}")
        raise

if __name__ == "__main__":
    test_summary_value_reference_safety()