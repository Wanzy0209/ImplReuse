import tensorflow as tf
import numpy as np

def test_repeat_elements_with_complex_arithmetic_logic():
    """
    Test case for tf.keras.backend.repeat_elements inspired by Issue 164385.
    
    The original issue involved a FloorDiv expression with specific constants 
    (24, 672, 2016, 22) simplifying incorrectly in a symbolic context.
    This test adapts that arithmetic logic to calculate the repetition factor
    for the repeat_elements API, ensuring the API handles the result correctly.
    """
    
    # Setup inputs mimicking the symbolic variables from the bug report
    # We use concrete values here to test the execution flow, as 
    # tf.keras.backend.repeat_elements operates on tensors, not SymPy symbols.
    # s14, s37, s46 are mapped to values that exercise the specific constants.
    s14 = tf.constant(2016, dtype=tf.int32)
    s37 = tf.constant(1, dtype=tf.int32)
    s46 = tf.constant(1, dtype=tf.int32)

    # Reproduce the logic from the bug report:
    # FloorDiv((24*s37 + 672)*(((s14*s46)//2016)) + 21, 22)
    # Using TensorFlow operations (tf.math.floordiv) to mirror the logic.
    
    # Step 1: Inner FloorDiv: (s14 * s46) // 2016
    inner_expr = tf.math.floordiv(s14 * s46, 2016)
    
    # Step 2: Middle expression: (24 * s37 + 672) * inner_expr
    middle_expr = (24 * s37 + 672) * inner_expr
    
    # Step 3: Numerator: middle_expr + 21
    numerator = middle_expr + 21
    
    # Step 4: Final FloorDiv to determine the repetition count
    # This corresponds to the 'rep' argument in repeat_elements
    rep_tensor = tf.math.floordiv(numerator, 22)
    
    # Convert to Python int as required by the specific API signature
    # (Note: The API docstring specifies 'rep: Python integer')
    rep_count = int(rep_tensor.numpy())

    # Create a tensor to repeat
    # Using a shape dimension (24) present in the bug report logic
    x = tf.ones((24, 1), dtype=tf.float32)
    
    # Perform the operation using the similar API
    # Axis 1 is chosen to repeat the columns
    result = tf.keras.backend.repeat_elements(x, rep_count, axis=1)
    
    # Assertions
    # With s14=2016, s37=1, s46=1:
    # inner = 2016 // 2016 = 1
    # middle = (24 + 672) * 1 = 696
    # numerator = 696 + 21 = 717
    # rep = 717 // 22 = 32
    # Expected shape: (24, 32)
    expected_shape = (24, 32)
    
    assert result.shape == expected_shape, \
        f"Shape mismatch. Expected {expected_shape}, got {result.shape}"
    
    # Verify values are preserved (all ones)
    assert np.all(result.numpy() == 1.0), "Value mismatch. Expected all ones."
    
    print("Test passed: repeat_elements handled complex arithmetic logic correctly.")

if __name__ == "__main__":
    test_repeat_elements_with_complex_arithmetic_logic()