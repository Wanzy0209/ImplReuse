import torch
import tensorflow as tf
import numpy as np

def test_tf_keras_ops_diff():
    """
    Adapted test case for tf.keras.ops.diff based on the PyTorch sparse.mm bug report.
    The original bug involved sparse and dense matrix multiplication causing a segfault.
    Here we verify the behavior of tf.keras.ops.diff with the same tensor structures.
    """
    
    # 1. Setup inputs based on the original PyTorch test case
    # Original a: torch.tensor([[1., 0, 2], [0, 3, 0]]).to_sparse()
    # Original b: torch.tensor([[0, 1.], [2, 0], [0, 0]])
    
    # Dense versions
    a_dense = tf.constant([[1., 0, 2], [0, 3, 0]])
    b_dense = tf.constant([[0, 1.], [2, 0], [0, 0]])

    # Sparse version of 'a' (mimicking .to_sparse())
    # Indices: [[0, 0], [0, 2], [1, 1]], Values: [1.0, 2.0, 3.0]
    a_sparse = tf.SparseTensor(
        indices=[[0, 0], [0, 2], [1, 1]], 
        values=[1.0, 2.0, 3.0], 
        dense_shape=(2, 3)
    )

    # 2. Test tf.keras.ops.diff on Dense Tensors
    # diff computes the difference between adjacent elements along the last axis by default.
    
    # Expected for a_dense: [[0-1, 2-0], [3-0, 0-3]] = [[-1, 2], [3, -3]]
    expected_diff_a = tf.constant([[-1., 2.], [3., -3.]])
    result_diff_a = tf.keras.ops.diff(a_dense)
    
    # Expected for b_dense: [[1-0], [0-2], [0-0]] = [[1], [-2], [0]]
    expected_diff_b = tf.constant([[1.], [-2.], [0.]])
    result_diff_b = tf.keras.ops.diff(b_dense)

    # Verify Dense Results
    assert np.allclose(result_diff_a.numpy(), expected_diff_a.numpy()), \
        f"Diff on dense A failed. Expected {expected_diff_a.numpy()}, got {result_diff_a.numpy()}"
    assert np.allclose(result_diff_b.numpy(), expected_diff_b.numpy()), \
        f"Diff on dense B failed. Expected {expected_diff_b.numpy()}, got {result_diff_b.numpy()}"

    # 3. Test tf.keras.ops.diff on Sparse Tensor
    # The original bug report highlighted a crash when handling sparse results.
    # We verify if tf.keras.ops.diff handles sparse inputs gracefully.
    
    try:
        result_diff_sparse = tf.keras.ops.diff(a_sparse)
        
        # Convert result to dense for verification
        result_diff_sparse_dense = tf.sparse.to_dense(result_diff_sparse)
        
        # The result should match the dense calculation
        assert np.allclose(result_diff_sparse_dense.numpy(), expected_diff_a.numpy()), \
            f"Diff on sparse A failed. Expected {expected_diff_a.numpy()}, got {result_diff_sparse_dense.numpy()}"
            
        print("Test passed: tf.keras.ops.diff handled both dense and sparse inputs correctly.")
        
    except Exception as e:
        print(f"Test encountered an exception with sparse input: {e}")
        raise

if __name__ == "__main__":
    test_tf_keras_ops_diff()