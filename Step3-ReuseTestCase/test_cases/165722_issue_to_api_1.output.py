import torch
import tensorflow as tf
import gc

def test_my_fact_reference_counting_on_reset():
    """
    Test case for tf.compat.v1.user_ops.my_fact based on the PyTorch 
    dangling pointer bug (Issue 165722).
    
    The original bug occurs when a C++ class stores a raw PyObject* 
    without reference counting, leading to a dangling pointer after 
    a reset operation (torch._dynamo.reset).
    
    This test verifies that the similar TensorFlow API handles object 
    lifecycles correctly across graph resets without crashing.
    """
    # Ensure we are in graph mode to allow resetting the graph state
    tf.compat.v1.disable_eager_execution()

    # 1. First execution: Create and run the op
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        # Call the API. Assuming the wrapper provided in the prompt:
        # def my_fact(): return _gen_user_ops.fact()
        # If the underlying C++ implementation stores raw pointers 
        # (similar to the PyTorch bug), this initializes that state.
        op = tf.compat.v1.user_ops.my_fact()
        
        # Execute to trigger internal state storage
        result_1 = sess.run(op)
        assert result_1 is not None

    # 2. Reset the environment
    # This corresponds to torch._dynamo.reset() in the original bug.
    # If the C++ implementation held a raw pointer to a graph object 
    # or tensor without Py_INCREF, that pointer becomes dangling here.
    tf.compat.v1.reset_default_graph()
    
    # Force garbage collection to attempt to free the invalidated objects
    gc.collect()

    # 3. Second execution: Access the op again
    # If the bug exists, the C++ layer might attempt to access the 
    # dangling pointer, causing a segmentation fault.
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        op = tf.compat.v1.user_ops.my_fact()
        result_2 = sess.run(op)
        assert result_2 is not None

    print("Test passed: No dangling pointer crash detected after reset.")

if __name__ == "__main__":
    test_my_fact_reference_counting_on_reset()