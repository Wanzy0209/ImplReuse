import torch
import tensorflow as tf
import numpy as np

def test_sparse_placeholder_dimensionality_handling():
    """
    Test case for tf.compat.v1.sparse_placeholder inspired by PyTorch Issue 164814.
    
    The original issue involves a dimensionality mismatch error when a scalar (0-d) tensor
    is processed, specifically related to strides and sizes during compilation.
    
    This test adapts the logic to TensorFlow by using tf.compat.v1.sparse_placeholder
    to feed a 1-D tensor, converting it to dense, and squeezing it to a scalar (0-D).
    This verifies that the dimensionality transition (1-D to 0-D) is handled correctly
    in the graph construction and execution, similar to the eager/compile divergence
    scenario in the original bug.
    """
    # Reset graph to ensure clean state
    tf.compat.v1.reset_default_graph()
    
    # Disable eager execution to simulate the "compile" aspect of the original bug
    tf.compat.v1.disable_eager_execution()

    # Define a sparse placeholder. 
    # In the PyTorch bug, the tensor leading to the error was size=(1,) before squeeze.
    # We use shape=[1] here to mimic that state.
    sp_placeholder = tf.compat.v1.sparse_placeholder(dtype=tf.int32, shape=[1])

    # Convert sparse tensor to dense to perform standard tensor operations
    dense_tensor = tf.sparse.to_dense(sp_placeholder)

    # Mimic the PyTorch logic: reshape to [1] then squeeze to scalar ().
    # PyTorch: var_node_1 = torch.reshape(var_node_2, [1])
    # PyTorch: var_node_0 = torch.squeeze(var_node_1) # size=()
    reshaped_tensor = tf.reshape(dense_tensor, [1])
    squeezed_tensor = tf.squeeze(reshaped_tensor)

    # Perform an operation on the scalar to ensure it is valid
    # PyTorch: result = var_node_0 * sentinel
    result = squeezed_tensor * 2

    with tf.compat.v1.Session() as sess:
        # Create a SparseTensorValue to feed the placeholder.
        # Represents a 1-D tensor with a single element (value 1).
        indices = np.array([[0]], dtype=np.int64)
        values = np.array([1], dtype=np.int32)
        shape = np.array([1], dtype=np.int64)
        sp_value = tf.compat.v1.SparseTensorValue(indices, values, shape)

        # Execute the graph
        output = sess.run(result, feed_dict={sp_placeholder: sp_value})

        # Assertions
        # The original bug failed because dimensionality of sizes (0) didn't match strides (1).
        # We assert that the output is indeed a scalar (0-d) and the value is correct.
        assert isinstance(output, (np.int32, np.int64)), f"Expected scalar type, got {type(output)}"
        assert output.shape == (), f"Expected scalar shape (), got {output.shape}"
        assert output == 2, f"Expected value 2, got {output}"
        
        print(" Test passed: Dimensionality handling for sparse placeholder is correct.")

if __name__ == "__main__":
    test_sparse_placeholder_dimensionality_handling()