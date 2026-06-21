import torch
import numpy as np
import unittest

# Attempt to import TensorFlow, handle environment errors gracefully
try:
    import tensorflow as tf
except ImportError as e:
    tf = None
    # Store the error to display in the skip message if needed
    _tf_import_error = str(e)
else:
    _tf_import_error = None

def test_initializer_int16_consistency():
    """
    Test case derived from PyTorch Issue 164704.
    Bug: Eager/Compile divergence with int16 scalars and item().
    Similar API: tf.compat.v1.global_variables_initializer (handles Eager vs Graph logic).
    
    This test verifies that tf.compat.v1.global_variables_initializer correctly handles
    variable initialization in both Eager and Graph modes, ensuring no divergence
    (unlike the PyTorch bug) when performing int16 scalar operations.
    """
    
    # Check if TensorFlow was successfully imported
    if tf is None:
        raise unittest.SkipTest(f"TensorFlow import failed due to environment issues (e.g., GLIBC): {_tf_import_error}")

    # 1. Test in Eager Mode (TF2 default)
    # In eager mode, global_variables_initializer returns a no-op.
    # Variables are initialized immediately upon creation.
    var_eager = tf.Variable(3, dtype=tf.int16, name='var_eager')
    
    # Call the Similar API
    init_op_eager = tf.compat.v1.global_variables_initializer()
    
    # Verify behavior: In eager, variables are initialized immediately.
    # The API returns a no-op, but we can check the variable value.
    assert var_eager.numpy() == 3
    
    # Perform the arithmetic logic from the PyTorch bug
    # PyTorch: var_node_1 = torch.div(var_node_2, var_node_5) where inputs are int16
    # Simplified: div(3, 1)
    val_eager = tf.divide(var_eager, tf.constant(1, dtype=tf.int16))
    
    # 2. Test in Graph Mode (using tf.compat.v1.Session to trigger the graph path of the API)
    # This mimics the "compile" mode in PyTorch where the bug occurred.
    # In this context, context.executing_eagerly() returns False.
    with tf.compat.v1.Session() as sess:
        var_graph = tf.Variable(3, dtype=tf.int16, name='var_graph')
        
        # Call the Similar API
        # It should return variables_initializer(global_variables()).
        init_op_graph = tf.compat.v1.global_variables_initializer()
        
        # Run the initializer
        sess.run(init_op_graph)
        
        # Perform the arithmetic
        val_graph_op = tf.divide(var_graph, tf.constant(1, dtype=tf.int16))
        
        # Execute
        result_graph = sess.run(val_graph_op)
        
    # 3. Compare results to check for divergence
    # The PyTorch bug resulted in a RuntimeError: expected int arg but got float.
    # We check if TF handles the type promotion and execution consistently.
    
    print(f"Eager result: {val_eager.numpy()}, dtype: {val_eager.dtype}")
    print(f"Graph result: {result_graph}, dtype: {result_graph.dtype}")
    
    # Check if the values are consistent
    np.testing.assert_almost_equal(val_eager.numpy(), result_graph)
    
    # Check if the types are consistent (tf.divide typically promotes int16 to float32/64)
    assert val_eager.dtype == result_graph.dtype

    print("Test passed: No divergence between Eager and Graph modes using global_variables_initializer.")

if __name__ == "__main__":
    # If running as a script, we need to handle the SkipTest exception manually
    # or run it via unittest.main(). To keep it simple and script-like:
    try:
        test_initializer_int16_consistency()
    except unittest.SkipTest as e:
        print(f"Test Skipped: {e}")