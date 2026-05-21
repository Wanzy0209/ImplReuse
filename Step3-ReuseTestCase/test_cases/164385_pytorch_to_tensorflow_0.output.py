import tensorflow as tf

def test_tf_strings_as_string_floor_div():
    """
    Adapted test case for tf.strings.as_string based on the FloorDiv issue.
    Original issue: FloorDiv((24*s37 + 672)*(((s14*s46)//2016)) + 21, 22) 
    generated a sympy Rational instead of maintaining FloorDiv structure.
    
    Here we verify that tf.strings.as_string correctly formats the result 
    of the equivalent floor division operation in TensorFlow.
    """
    
    # Create concrete tensors to represent the symbolic variables
    # Using values that exercise the logic (e.g., s14 is a multiple of 2016)
    s14 = tf.constant(4032, dtype=tf.int32) 
    s37 = tf.constant(1, dtype=tf.int32)
    s46 = tf.constant(1, dtype=tf.int32)

    print("Testing tf.strings.as_string with complex floor division expression...")

    # Build the numerator expression step by step using TensorFlow ops
    # Original logic: inner_expr = FloorDiv(s14 , 2016)
    # Note: The code block in the issue used s14 // 2016, though the title included s46.
    # We follow the executable code block logic provided in the issue.
    inner_expr = tf.math.floordiv(s14, 2016)
    
    # Original logic: middle_expr = (24 * s37 + 672) * inner_expr
    middle_expr = (24 * s37 + 672) * inner_expr
    
    # Original logic: numerator = middle_expr + 21
    numerator = middle_expr + 21
    
    # Original logic: result = FloorDiv(numerator, 22)
    # In TensorFlow, we use tf.math.floordiv for integer floor division
    floor_div_result = tf.math.floordiv(numerator, 22)
    
    print(f"FloorDiv result (Tensor): {floor_div_result}")
    
    # Apply the Similar API: tf.strings.as_string
    # This converts the numeric result to a string representation.
    string_result = tf.strings.as_string(floor_div_result)
    
    print(f"String result: {string_result}")
    print(f"String result (numpy): {string_result.numpy()}")

    # Verify the result
    # Calculation trace:
    # s14=4032, s37=1
    # inner = 4032 // 2016 = 2
    # middle = (24 + 672) * 2 = 696 * 2 = 1392
    # num = 1392 + 21 = 1413
    # res = 1413 // 22 = 64 (since 22 * 64 = 1408)
    expected_string = b'64'
    
    assert string_result.numpy()[0] == expected_string, \
        f"Expected {expected_string}, got {string_result.numpy()[0]}"
        
    print("Test passed.")

if __name__ == "__main__":
    test_tf_strings_as_string_floor_div()