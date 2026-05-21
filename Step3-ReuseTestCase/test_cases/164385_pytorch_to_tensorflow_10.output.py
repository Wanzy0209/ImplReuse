import tensorflow as tf
import numpy as np

def test_tf_experimental_numpy_add_complex_expression():
    """
    Adapted test case for tf.experimental.numpy.add based on the FloorDiv bug report.
    
    The original issue involved FloorDiv generating a sympy Rational (float-like behavior)
    instead of maintaining integer division semantics. This test verifies that 
    tf.experimental.numpy.add maintains integer types correctly within a similar 
    complex arithmetic expression structure.
    """
    
    print("Testing tf.experimental.numpy.add with complex expression...")

    # Create variables as TensorFlow tensors (integers)
    # Using specific positive integers to simulate the symbolic variables
    s14 = tf.constant(4032, dtype=tf.int32)  # 4032 // 2016 = 2
    s37 = tf.constant(1, dtype=tf.int32)
    s46 = tf.constant(1, dtype=tf.int32)

    # Build the numerator expression step by step
    # Original: inner_expr = FloorDiv(s14 , 2016)
    # Using tf.experimental.numpy.floor_divide to match the logic
    inner_expr = tf.experimental.numpy.floor_divide(s14, 2016)

    # Original: middle_expr = (24 * s37 + 672) * inner_expr
    # We explicitly use tf.experimental.numpy.add here as requested by the target API
    term1 = 24 * s37
    term2 = 672
    sum_term = tf.experimental.numpy.add(term1, term2)
    middle_expr = sum_term * inner_expr

    # Original: numerator = middle_expr + 21
    # We explicitly use tf.experimental.numpy.add here
    numerator = tf.experimental.numpy.add(middle_expr, 21)

    denominator = 22

    print(f"Numerator: {numerator}")
    print(f"Denominator: {denominator}")

    # Original: result = FloorDiv(numerator, denominator)
    result = tf.experimental.numpy.floor_divide(numerator, denominator)

    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    print(f"Result dtype: {result.dtype}")

    # Verify the result
    # Calculation trace:
    # inner_expr = 4032 // 2016 = 2
    # sum_term = (24 * 1) + 672 = 696
    # middle_expr = 696 * 2 = 1392
    # numerator = 1392 + 21 = 1413
    # result = 1413 // 22 = 64
    expected_value = 64
    
    assert result == expected_value, f"Expected value {expected_value}, but got {result}"
    
    # The original bug resulted in a Rational (float-like). 
    # We assert that the result remains an integer type.
    assert result.dtype == tf.int32, f"Expected dtype int32, but got {result.dtype}"
    
    print("Test passed: tf.experimental.numpy.add preserved integer types correctly.")

if __name__ == "__main__":
    test_tf_experimental_numpy_add_complex_expression()