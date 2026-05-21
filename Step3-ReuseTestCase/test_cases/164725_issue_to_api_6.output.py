import torch
import tensorflow as tf
import numpy as np

def test_linear_operator_composition_scalar_divergence():
    """
    Test case for tf.linalg.LinearOperatorComposition inspired by 
    PyTorch Issue 164725 (DDE when calling .item()).
    
    This test checks for divergence between eager and compiled (tf.function) modes
    when extracting a scalar value from a composed operator chain and using it
    in a computation with a gradient-tracking variable (sentinel).
    """
    
    # Setup: Create operators to mimic the tensor manipulation chain
    # Using float32 as LinearOperators typically require numeric types
    matrix_a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    matrix_b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    
    op_a = tf.linalg.LinearOperatorFullMatrix(matrix_a)
    op_b = tf.linalg.LinearOperatorFullMatrix(matrix_b)
    
    # Compose operators (mimics the reshape/select/squeeze chain complexity)
    composed_op = tf.linalg.LinearOperatorComposition([op_a, op_b])
    
    # Sentinel variable to ensure gradient computation (mimics requires_grad=True)
    sentinel = tf.Variable(1.0)

    def computation_logic(op, sent):
        # Perform a reduction to a scalar (mimics .item() on a scalar tensor)
        # trace() returns a scalar tensor representing the sum of diagonal elements
        scalar_val = op.trace()
        
        # Use the scalar in a computation (mimics result = var_node_0 * sentinel)
        result = scalar_val * sent
        return result

    # 1. Run in Eager Mode
    with tf.GradientTape() as tape_eager:
        result_eager = computation_logic(composed_op, sentinel)
    grad_eager = tape_eager.gradient(result_eager, sentinel)

    # 2. Run in Compiled Mode (tf.function mimics torch.compile)
    compiled_logic = tf.function(computation_logic)
    
    with tf.GradientTape() as tape_compiled:
        result_compiled = compiled_logic(composed_op, sentinel)
    grad_compiled = tape_compiled.gradient(result_compiled, sentinel)

    # Assertions to check for divergence
    # Check if the scalar results match
    assert np.allclose(result_eager.numpy(), result_compiled.numpy()), \
        f"Divergence in scalar output: Eager={result_eager.numpy()}, Compiled={result_compiled.numpy()}"
    
    # Check if the gradients match
    assert np.allclose(grad_eager.numpy(), grad_compiled.numpy()), \
        f"Divergence in gradient: Eager={grad_eager.numpy()}, Compiled={grad_compiled.numpy()}"

    print(' Eager and Compiled results match for LinearOperatorComposition.')

if __name__ == "__main__":
    test_linear_operator_composition_scalar_divergence()