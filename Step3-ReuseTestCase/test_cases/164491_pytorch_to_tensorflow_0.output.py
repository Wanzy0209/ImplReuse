import tensorflow as tf
import numpy as np

def test_tf_keras_backend_switch_int_tensors():
    """
    Adapted from PyTorch issue #164491 regarding _int_mm and _scaled_mm 
    performance/errors with row-major matrices.

    The original issue highlights that specific internal matrix multiplication 
    functions (_int_mm, _scaled_mm) behave poorly (errors/slow) when the 
    Right-Hand Side (RHS) matrix is in row-major layout compared to column-major.

    Since tf.keras.backend.switch is a conditional selection API and not a 
    matrix multiplication kernel, we adapt the test to verify that switch 
    correctly handles integer tensors (referencing the _int_mm context) and 
    properly selects between two distinct tensor states (representing the 
    choice between row-major and column-major data paths).
    """
    # Create integer tensors to mimic the _int_mm context
    # State 1: Represents a "Column-major" compatible data state
    tensor_col_major = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
    
    # State 2: Represents a "Row-major" compatible data state
    # (Using distinct values to verify selection logic)
    tensor_row_major = tf.constant([[7, 8, 9], [10, 11, 12]], dtype=tf.int32)

    # Test Case 1: Condition is True (Select Column-major state)
    condition_true = tf.constant(True)
    result_true = tf.keras.backend.switch(condition_true, tensor_col_major, tensor_row_major)
    
    # Assert that the result matches the 'then_expression'
    assert tf.reduce_all(tf.equal(result_true, tensor_col_major)).numpy(), \
        "Switch failed to select the 'then_expression' (column-major state) when condition is True."

    # Test Case 2: Condition is False (Select Row-major state)
    condition_false = tf.constant(False)
    result_false = tf.keras.backend.switch(condition_false, tensor_col_major, tensor_row_major)
    
    # Assert that the result matches the 'else_expression'
    assert tf.reduce_all(tf.equal(result_false, tensor_row_major)).numpy(), \
        "Switch failed to select the 'else_expression' (row-major state) when condition is False."

    # Test Case 3: Verify shape preservation (critical for matrix operations)
    # The original bug involved implicit transposes; we ensure switch preserves shape.
    assert result_true.shape == tensor_col_major.shape, \
        "Switch altered the shape of the selected tensor."
    assert result_false.shape == tensor_row_major.shape, \
        "Switch altered the shape of the selected tensor."

if __name__ == "__main__":
    test_tf_keras_backend_switch_int_tensors()
    print("Test passed successfully.")