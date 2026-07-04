import tensorflow as tf
import numpy as np

def test_diagonal_with_complex_expression():
    """
    Test tf.experimental.numpy.diagonal with a complex tensor expression
    derived from the FloorDiv bug report logic.
    
    The original bug report involves a complex symbolic expression:
    FloorDiv((24*s37 + 672)*(((s14*s46)//2016)) + 21, 22)
    and checks if the result maintains the correct type (FloorDiv vs Mul).
    
    This test adapts that logic to tf.experimental.numpy.diagonal by:
    1. Constructing a complex tensor expression using TensorFlow operations.
    2. Using that expression to build a matrix input for diagonal.
    3. Verifying that diagonal extracts the correct values and preserves the tensor type.
    """

    # Check if the required API exists in the current TensorFlow version
    if not hasattr(tf.experimental, 'numpy'):
        print("Skipping test: tf.experimental.numpy is not available in this TensorFlow version.")
        return

    print("Testing tf.experimental.numpy.diagonal with complex tensor expression...")

    # Create symbolic variables (as TensorFlow constants/tensors)
    # s14, s37, s46 are integer symbols in the bug report
    s14 = tf.constant([2016, 4032, 6048], dtype=tf.int32)
    s37 = tf.constant([1, 2, 3], dtype=tf.int32)
    s46 = tf.constant([1, 1, 1], dtype=tf.int32)

    # Build the numerator expression step by step
    # Original: inner_expr = FloorDiv(s14 , 2016)
    inner_expr = tf.math.floordiv(s14, 2016)
    
    # Original: middle_expr = (24 * s37 + 672) * inner_expr
    middle_expr = (24 * s37 + 672) * inner_expr
    
    # Original: numerator = middle_expr + 21
    numerator = middle_expr + 21

    print(f"Numerator expression result: {numerator}")

    # Create a matrix input for diagonal using the complex expression
    # We construct a diagonal matrix where the diagonal elements are the result of our complex expression
    # This mirrors the structure of passing a complex expression to an operator.
    matrix_input = tf.linalg.diag(numerator)
    
    print(f"Matrix Input:\n{matrix_input}")

    # Call the similar API: tf.experimental.numpy.diagonal
    # The bug report checks FloorDiv(numerator, 22).
    # We check diagonal(matrix_input, offset=0).
    result = tf.experimental.numpy.diagonal(matrix_input, offset=0)
    
    print(f"Diagonal result: {result}")
    print(f"Diagonal result type: {type(result)}")
    print(f"Diagonal result dtype: {result.dtype}")

    # Assertions
    # 1. Check that the result matches the expected complex expression result
    expected = numerator
    assert tf.reduce_all(result == expected).numpy(), \
        f"Value mismatch. Expected {expected}, got {result}"
    
    # 2. Check type preservation (mirroring the FloorDiv type check in the bug report)
    # The bug report ensures the result is a FloorDiv object, not a Mul.
    # Here we ensure the result is a Tensor and not simplified to a different type unexpectedly.
    assert isinstance(result, tf.Tensor), "Result should be a Tensor"
    assert result.dtype == tf.int32, "Result dtype should be preserved as int32"

    # 3. Test with a non-zero offset to exercise the 'moveaxis' logic in the similar API implementation
    # The similar API implementation has specific logic for offsets and axis manipulation.
    result_offset = tf.experimental.numpy.diagonal(matrix_input, offset=1)
    expected_offset = tf.constant([0, 0], dtype=tf.int32) # Off-diagonal elements are 0
    assert tf.reduce_all(result_offset == expected_offset).numpy(), \
        f"Offset mismatch. Expected {expected_offset}, got {result_offset}"

    print("Test passed.")

if __name__ == "__main__":
    test_diagonal_with_complex_expression()