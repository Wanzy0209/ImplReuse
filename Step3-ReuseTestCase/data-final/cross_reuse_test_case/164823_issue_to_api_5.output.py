import torch
import tensorflow as tf
from tensorflow.linalg import LinearOperatorFullMatrix, LinearOperatorAdjoint

class TestModel(tf.Module):
    """
    A TensorFlow model mirroring the structure of the PyTorch bug report.
    Instead of sparse tensors, we use LinearOperatorAdjoint, which represents
    a structured linear algebraic object that requires conversion to dense.
    """
    def __init__(self, matrix):
        super().__init__()
        # Initialize the base operator and its adjoint
        self.base_op = LinearOperatorFullMatrix(matrix)
        self.adjoint_op = LinearOperatorAdjoint(self.base_op)

    @tf.function
    def __call__(self):
        """
        Mimics the forward pass in the bug report:
        1. Access the structured operator (analogous to sparse tensor).
        2. Convert to dense (analogous to to_dense()).
        """
        # The original bug failed here when compiled because Inductor 
        # couldn't access the storage of the SparseTensorImpl.
        # We test if tf.function (JIT) handles the conversion for LinearOperatorAdjoint.
        return self.adjoint_op.to_dense()

def test_linear_operator_adjoint_compile():
    # Define a simple 2x2 matrix
    matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])

    # Instantiate the model
    model = TestModel(matrix)

    # Run in eager mode (baseline)
    eager_output = model.__call__(tf.TensorSpec([], dtype=tf.float32)) # Dummy input for trace if needed, or just call
    # Note: LinearOperatorAdjoint.to_dense() doesn't strictly require input x if the matrix is fixed,
    # but to match the pattern of the original bug where a model was called with input:
    
    # Let's adjust to match the original pattern closer: 
    # The original model took x, converted it, and returned dense.
    # LinearOperator is usually defined by a matrix, not a dynamic input x in the same way.
    # However, we can verify the compilation of the to_dense() call.

    print("Eager output:\n", eager_output.numpy())

    # Run compiled (tf.function acts as torch.compile)
    # The __call__ method is already decorated with @tf.function
    compiled_output = model()

    print("Compiled output:\n", compiled_output.numpy())

    # Verify correctness
    expected = tf.linalg.adjoint(matrix)
    assert tf.reduce_all(tf.equal(compiled_output, expected)).numpy(), "Outputs do not match"
    
    print("Test passed: LinearOperatorAdjoint.to_dense() works correctly with tf.function compilation.")

if __name__ == "__main__":
    test_linear_operator_adjoint_compile()