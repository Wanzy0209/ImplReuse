import torch
import tensorflow as tf

def test_floor_div_complex_expression():
    """
    Adapted test case for TensorFlow based on the FloorDiv bug report.
    The original issue involved symbolic simplification (Sympy) where FloorDiv 
    was incorrectly converted to a Mul with a Rational.
    
    Since tf.concat is semantically different from torch.div (floor division),
    we use tf.math.floordiv to preserve the core logic of the bug reproduction.
    """
    print("Testing tf.math.floordiv with complex expression...")

    # Create integer tensors to simulate symbolic variables
    # We use specific values to verify the arithmetic logic
    s14 = tf.constant(4032, dtype=tf.int32)   # 4032 // 2016 = 2
    s37 = tf.constant(10, dtype=tf.int32)
    # s46 is present in the title but not the code snippet logic, omitted for fidelity to snippet.

    # Build the numerator expression step by step
    # Original: FloorDiv(s14, 2016)
    inner_expr = tf.math.floordiv(s14, 2016)
    
    # Original: (24 * s37 + 672) * inner_expr
    middle_expr = (24 * s37 + 672) * inner_expr
    
    # Original: middle_expr + 21
    numerator = middle_expr + 21
    
    denominator = 22

    print(f"Numerator value: {numerator.numpy()}")
    print(f"Denominator value: {denominator}")

    # Create the FloorDiv expression
    # Original: FloorDiv(numerator, denominator)
    result = tf.math.floordiv(numerator, denominator)
    
    print(f"tf.math.floordiv result: {result.numpy()}")
    print(f"Result type: {type(result)}")

    # Verify the result
    # Calculation:
    # inner = 4032 // 2016 = 2
    # middle = (24*10 + 672) * 2 = (240 + 672) * 2 = 912 * 2 = 1824
    # numerator = 1824 + 21 = 1845
    # result = 1845 // 22 = 83
    expected = 83
    
    assert result.numpy() == expected, f"Expected {expected}, but got {result.numpy()}"
    print("Test passed.")

if __name__ == "__main__":
    test_floor_div_complex_expression()