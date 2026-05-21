import tensorflow as tf

def test_tf_math_add_complex_expression():
    """
    Adapted test case for tf.compat.v1.math.add based on the FloorDiv issue.
    The original issue involved FloorDiv simplifying to a Rational multiplication.
    Here we verify that tf.compat.v1.math.add handles complex expressions correctly
    without unexpected simplification or type coercion issues, mirroring the structure
    of the original test case.
    """
    
    # Disable eager execution to mimic v1 graph behavior if necessary, 
    # though tf.function is the modern equivalent. 
    # For this specific API (compat.v1), we often work with tensors directly.
    
    print("Testing tf.compat.v1.math.add with complex expression...")

    # Create symbolic-like variables (using placeholders or tensors)
    # In TensorFlow, we don't have sympy symbols, but we can use tf.Variable or tf.constant
    # to represent the inputs s14, s37, s46.
    
    # Note: The original bug was about FloorDiv((24*s37 + 672)*(((s14*s46)//2016)) + 21, 22)
    # generating a sympy Rational. 
    # Since we are testing tf.math.add, we adapt the logic to an addition context
    # while preserving the complexity of the operands.
    
    # Define inputs
    s14 = tf.constant(4032, dtype=tf.int32) # s14 // 2016 = 2
    s37 = tf.constant(10, dtype=tf.int32)
    s46 = tf.constant(1, dtype=tf.int32)
    
    # Build the expression components
    # Original: (24*s37 + 672)
    term1 = (24 * s37) + 672
    
    # Original: ((s14*s46)//2016)
    # Note: tf.math.floordiv is the TF equivalent of //
    inner_div = tf.math.floordiv(s14 * s46, 2016)
    
    # Original: (24*s37 + 672) * inner_expr
    middle_expr = term1 * inner_div
    
    # Original: middle_expr + 21
    # This is where we apply the target API: tf.compat.v1.math.add
    # We are adding 21 to the complex middle expression.
    add_result = tf.compat.v1.math.add(middle_expr, 21)
    
    print(f"Term1 (24*s37 + 672): {term1.numpy()}")
    print(f"Inner Div ((s14*s46)//2016): {inner_div.numpy()}")
    print(f"Middle Expr (Term1 * Inner Div): {middle_expr.numpy()}")
    print(f"Add Result (Middle Expr + 21): {add_result.numpy()}")
    
    # Verify the result
    # s14=4032, s37=10, s46=1
    # term1 = 240 + 672 = 912
    # inner_div = 4032 // 2016 = 2
    # middle_expr = 912 * 2 = 1824
    # add_result = 1824 + 21 = 1845
    
    expected_result = 1845
    
    # Check if the result matches the expected integer value
    # The original bug was about type conversion (to Rational), so we check type and value.
    assert add_result.numpy() == expected_result, f"Expected {expected_result}, got {add_result.numpy()}"
    assert add_result.dtype == tf.int32, f"Expected dtype int32, got {add_result.dtype}"
    
    print("Test passed: tf.compat.v1.math.add handled the complex expression correctly.")

if __name__ == "__main__":
    test_tf_math_add_complex_expression()