import tensorflow as tf
import numpy as np

def test_minimum_with_complex_integer_expression():
    """
    Test tf.raw_ops.Minimum with complex integer expressions to ensure 
    type preservation, mirroring the FloorDiv -> Rational issue.
    """
    print("Testing Minimum with complex integer expression...")

    # Create integer tensors (mimicking symbolic variables s14, s37, s46)
    # Using values that allow for the arithmetic operations in the original bug
    s14 = tf.constant(4032, dtype=tf.int32)  # s14 // 2016 = 2
    s37 = tf.constant(10, dtype=tf.int32)
    s46 = tf.constant(1, dtype=tf.int32)

    # Build the complex expression similar to the original bug report:
    # Original: FloorDiv((24*s37 + 672)*(((s14*s46)//2016)) + 21, 22)
    # We adapt this to create two inputs for Minimum.
    
    # Inner expression: (s14 * s46) // 2016
    inner_expr = tf.math.floordiv(s14 * s46, 2016)
    
    # Middle expression: (24 * s37 + 672) * inner_expr
    middle_expr = (24 * s37 + 672) * inner_expr
    
    # Input 1: middle_expr + 21
    input_x = middle_expr + 21
    
    # Input 2: A constant 22 (the denominator from the original bug)
    input_y = tf.constant(22, dtype=tf.int32)

    print(f"Input X: {input_x}")
    print(f"Input Y: {input_y}")

    # Perform the Minimum operation
    # The original issue was FloorDiv turning into a Rational (float-like).
    # We check if Minimum preserves the integer type.
    result = tf.raw_ops.Minimum(x=input_x, y=input_y)
    
    print(f"Minimum result: {result}")
    print(f"Result dtype: {result.dtype}")

    # Assertions
    # 1. Check that the result is correct (min of 921 and 22 is 22)
    # middle_expr = (240 + 672) * 2 = 1824
    # input_x = 1824 + 21 = 1845
    # input_y = 22
    # min(1845, 22) = 22
    assert result.numpy() == 22, f"Expected 22, got {result.numpy()}"
    
    # 2. Check that the dtype is preserved as int32
    # This mirrors the original bug where FloorDiv (int) became Mul (Rational/float).
    assert result.dtype == tf.int32, f"Expected dtype tf.int32, got {result.dtype}"

    print("Test passed: Minimum preserved integer type and value.")

if __name__ == "__main__":
    test_minimum_with_complex_integer_expression()