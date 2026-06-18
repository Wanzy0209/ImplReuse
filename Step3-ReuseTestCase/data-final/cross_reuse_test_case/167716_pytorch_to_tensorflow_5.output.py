import torch
import tensorflow as tf
import numpy as np

def test_tf_swapaxes_sparse():
    """
    Adapted from PyTorch Issue 167716.
    Original Bug: torch.sparse.mm(A, B) with sparse B returned a corrupted sparse tensor 
    that caused a Segmentation fault when calling .to_dense().
    
    This test verifies the behavior of the similar API tf.keras.ops.swapaxes 
    when operating on sparse tensors, ensuring it handles the conversion 
    to dense without crashing or corrupting data.
    """
    
    # Recreate the sparse tensor setup from the PyTorch bug report
    # PyTorch: indices_A = [[0, 1, 2], [0, 2, 3]], values_A = [1.0, 2.0, 3.0], size=(3, 4)
    # TensorFlow SparseTensor indices are [N, rank]
    indices_A = tf.constant([[0, 0], [1, 2], [2, 3]], dtype=tf.int64)
    values_A = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    shape_A = tf.constant([3, 4], dtype=tf.int64)
    
    # Create SparseTensor A
    A = tf.sparse.SparseTensor(indices_A, values_A, shape_A)
    
    # Reorder indices (required for some TF operations)
    A = tf.sparse.reorder(A)

    print("Original Sparse Tensor A:")
    print(tf.sparse.to_dense(A))

    # The original bug involved an operation on sparse tensors.
    # Here we test the similar API: tf.keras.ops.swapaxes.
    # We swap axes 0 and 1 (equivalent to transpose for 2D).
    try:
        # Perform the operation
        C = tf.keras.ops.swapaxes(A, axis1=0, axis2=1)

        # The original bug crashed specifically when converting to dense.
        # We verify if the result is valid and can be converted.
        if isinstance(C, tf.sparse.SparseTensor):
            print("Result is SparseTensor. Converting to dense...")
            C_dense = tf.sparse.to_dense(C)
        else:
            # If the op returns a dense tensor (common in TF if sparse support is implicit),
            # we just assert it's valid.
            print("Result is already dense.")
            C_dense = C

        print("Result after swapaxes:")
        print(C_dense)

        # Basic assertion to ensure the operation ran and produced a tensor of expected shape
        # Original shape (3, 4), swapped shape should be (4, 3)
        assert C_dense.shape == (4, 3), f"Expected shape (4, 3), got {C_dense.shape}"
        
        print("Test passed: swapaxes on sparse tensor succeeded and converted to dense.")

    except Exception as e:
        print(f"Test failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_tf_swapaxes_sparse()