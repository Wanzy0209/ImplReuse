import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to use tf.compat.v1.placeholder (TF 1.x style)
tf.compat.v1.disable_eager_execution()

def test_sparse_matmul_to_dense():
    # Define data matching the PyTorch bug report
    # Matrix A (3x4)
    indices_A = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    values_A = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape_A = np.array([3, 4], dtype=np.int64)

    # Matrix B (4x2)
    indices_B = np.array([[0, 1, 2, 3], [0, 1, 1, 2]], dtype=np.int64)
    values_B = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    shape_B = np.array([4, 2], dtype=np.int64)

    # Create placeholders using the specified similar API: tf.compat.v1.sparse_placeholder
    # This mimics the creation of sparse tensors A and B in PyTorch
    A = tf.compat.v1.sparse_placeholder(tf.float32, shape=shape_A, name="A_sparse")
    B = tf.compat.v1.sparse_placeholder(tf.float32, shape=shape_B, name="B_sparse")

    # Perform sparse matrix multiplication
    # PyTorch equivalent: torch.sparse.mm(A, B)
    # TensorFlow equivalent: tf.sparse.matmul(A, B)
    # Note: tf.sparse.matmul supports both sparse x dense and sparse x sparse
    C_sparse = tf.sparse.matmul(A, B)

    # Convert the result to dense
    # PyTorch equivalent: C.to_dense()
    # TensorFlow equivalent: tf.sparse.to_dense(C_sparse)
    C_dense = tf.sparse.to_dense(C_sparse, name="C_dense")

    # Execute the graph
    with tf.compat.v1.Session() as sess:
        try:
            # Feed the sparse tensor data using the tuple format (indices, values, shape)
            result = sess.run(C_dense, feed_dict={
                A: (indices_A, values_A, shape_A),
                B: (indices_B, values_B, shape_B)
            })

            print("TensorFlow Sparse MatMul + to_dense succeeded.")
            print("Result:\n", result)

            # Verify the shape of the output
            # A is (3,4), B is (4,2), so C should be (3,2)
            assert result.shape == (3, 2), f"Expected shape (3, 2), got {result.shape}"
            
            # Verify specific values based on the inputs
            # A[0,0]=1, A[1,2]=2, A[2,3]=3
            # B[0,0]=4, B[1,1]=5, B[2,1]=6, B[3,2]=7
            # C[0,0] = A[0,0]*B[0,0] = 1*4 = 4
            # C[1,1] = A[1,2]*B[2,1] = 2*6 = 12
            # C[2,2] = A[2,3]*B[3,2] = 3*7 = 21
            assert result[0, 0] == 4.0, "Value mismatch at [0,0]"
            assert result[1, 1] == 12.0, "Value mismatch at [1,1]"
            assert result[2, 2] == 21.0, "Value mismatch at [2,2]"

        except Exception as e:
            print(f"Test failed with error: {e}")
            raise

if __name__ == "__main__":
    test_sparse_matmul_to_dense()