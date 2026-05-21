import torch
import tensorflow as tf

def test_sparse_ops_in_name_scope():
    """
    Adapts the PyTorch sparse/dense conversion test to use tf.compat.v1.name_scope.
    The original bug involved torch.compile failing with sparse tensors.
    Here we verify that the sparse operations work within the TensorFlow name_scope
    in both eager and compiled (graph) modes.
    """
    # Create a random dense tensor
    x = tf.random.normal((10, 10))

    # Define the model logic using the target API: tf.compat.v1.name_scope
    def run_logic(x_input):
        with tf.compat.v1.name_scope("sparse_ops_scope"):
            # Convert to sparse (equivalent to x.to_sparse())
            x_sparse = tf.sparse.from_dense(x_input)
            
            # Perform operation (equivalent to x_sparse * 2)
            result_sparse = x_sparse * 2
            
            # Convert back to dense (equivalent to result.to_dense())
            result_dense = tf.sparse.to_dense(result_sparse)
            return result_dense

    # 1. Test in Eager mode
    print("Eager output:", run_logic(x))

    # 2. Test in Compiled mode (tf.function is the TF equivalent to torch.compile)
    # This verifies that the name_scope and sparse ops work correctly when traced/compiled.
    compiled_model = tf.function(run_logic)
    print("Compiled output:", compiled_model(x))

if __name__ == "__main__":
    test_sparse_ops_in_name_scope()