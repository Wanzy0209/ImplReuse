import tensorflow as tf
import numpy as np

# Test case for tf.linalg.LinearOperatorComposition
# Adapted from the logic of the PyTorch index_add_ complex tensor bug.
# The original bug highlighted an issue where the imaginary part of complex 
# tensors was ignored/dropped on a specific backend. This test verifies that
# LinearOperatorComposition correctly preserves and processes the imaginary
# components of complex tensors.

def test_linear_operator_composition_complex():
    tf.random.set_seed(0)
    
    # Define shapes for the operators (2x2 matrices)
    shape = (2, 2)

    # Create complex tensors for the operators, explicitly separating real and imag
    # to mirror the construction in the original bug report.
    real_1 = tf.random.normal(shape, dtype=tf.float32)
    imag_1 = tf.random.normal(shape, dtype=tf.float32)
    tensor_1 = tf.complex(real_1, imag_1)

    real_2 = tf.random.normal(shape, dtype=tf.float32)
    imag_2 = tf.random.normal(shape, dtype=tf.float32)
    tensor_2 = tf.complex(real_2, imag_2)

    # Create LinearOperators
    op_1 = tf.linalg.LinearOperatorFullMatrix(tensor_1)
    op_2 = tf.linalg.LinearOperatorFullMatrix(tensor_2)

    # Compose operators: op_composed(x) = op_1(op_2(x))
    op_composed = tf.linalg.LinearOperatorComposition([op_1, op_2])

    # Create a complex input vector
    x_real = tf.random.normal((2, 1), dtype=tf.float32)
    x_imag = tf.random.normal((2, 1), dtype=tf.float32)
    x = tf.complex(x_real, x_imag)

    # Apply the composed operator
    result_composed = op_composed.matmul(x)

    # Calculate expected result using standard matmul (ground truth)
    result_expected = tf.linalg.matmul(tensor_1, tf.linalg.matmul(tensor_2, x))

    # Check imaginary parts (mirroring the bug report's check for dropped imag values)
    composed_imag_sum = tf.reduce_sum(tf.abs(tf.math.imag(result_composed))).numpy()
    expected_imag_sum = tf.reduce_sum(tf.abs(tf.math.imag(result_expected))).numpy()

    print("Composed imag sum:", composed_imag_sum)
    print("Expected imag sum:", expected_imag_sum)

    # Calculate max absolute difference
    max_diff = tf.reduce_max(tf.abs(result_composed - result_expected)).numpy()
    print("max abs diff:", max_diff)

    # Assertions to ensure correctness
    # 1. Ensure imaginary part is not zeroed out (the specific bug symptom)
    assert composed_imag_sum > 1e-5, "Imaginary part was incorrectly zeroed out."
    # 2. Ensure numerical consistency with expected calculation
    assert max_diff < 1e-5, f"Result differs from expected by {max_diff}"

if __name__ == "__main__":
    test_linear_operator_composition_complex()