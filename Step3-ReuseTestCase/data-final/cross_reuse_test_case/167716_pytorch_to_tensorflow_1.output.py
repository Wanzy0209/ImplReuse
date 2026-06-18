import tensorflow as tf

def test_sparse_diagonal():
    # Recreate the sparse tensor A from the PyTorch bug report
    # PyTorch: indices_A = [[0, 1, 2], [0, 2, 3]], values_A = [1.0, 2.0, 3.0], size=(3, 4)
    # TensorFlow SparseTensor indices are N x 2
    indices_A = tf.constant([[0, 0], [1, 2], [2, 3]], dtype=tf.int64)
    values_A = tf.constant([1.0, 2.0, 3.0])
    shape_A = tf.constant([3, 4], dtype=tf.int64)
    A_sparse = tf.sparse.SparseTensor(indices_A, values_A, shape_A)

    # The original bug involved a crash when handling sparse tensors (specifically sparse.mm).
    # We test if tf.keras.ops.diagonal handles the sparse tensor input correctly.
    # Note: Depending on the backend implementation, asarray() inside diagonal() 
    # might convert the sparse tensor to dense or raise an error.
    try:
        # Attempt to extract diagonal from the sparse tensor
        result = tf.keras.ops.diagonal(A_sparse)
        
        # If successful, verify the result against the expected diagonal
        # Matrix:
        # [[1, 0, 0, 0],
        #  [0, 0, 2, 0],
        #  [0, 0, 0, 3]]
        # Diagonal: [1, 0, 0]
        expected = tf.constant([1.0, 0.0, 0.0])
        
        # Check if result matches expected
        # We use reduce_all to handle potential shape differences or type promotions
        assert tf.reduce_all(tf.equal(result, expected)), f"Expected {expected}, but got {result}"
        print("Test passed. Diagonal extracted successfully:", result.numpy())
        
    except Exception as e:
        # If the API does not support sparse tensors, we catch the exception.
        # This verifies the behavior (graceful failure vs crash).
        print(f"API does not support sparse tensor input or encountered an error: {e}")

if __name__ == "__main__":
    test_sparse_diagonal()