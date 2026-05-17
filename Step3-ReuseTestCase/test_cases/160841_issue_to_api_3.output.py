import tensorflow as tf
import numpy as np

def test_tensor_diag_part_bfloat16_fix():
    """
    Test case adapted from Issue 160841.
    
    The original issue reported that running a model with 'auto' dtype on MacOS
    returned garbage, but explicitly setting the dtype to 'bf16' fixed it.
    
    This test verifies that the similar API, tf.compat.v1.linalg.tensor_diag_part,
    handles the 'fix' data type (bfloat16) correctly without returning garbage values.
    """
    # Create a sample diagonal matrix input
    # Using float32 initially to simulate standard data
    input_data = np.array([[1.5, 0.0, 0.0],
                           [0.0, 2.5, 0.0],
                           [0.0, 0.0, 3.5]], dtype=np.float32)

    # Apply the "fix" mentioned in the bug report: Cast to bfloat16
    # In the original issue: "Changing data type to bf16 fixes the problem."
    input_tensor_bf16 = tf.cast(input_data, tf.bfloat16)

    # Execute the similar API operation
    # This corresponds to the model generation step in the original issue,
    # where the tensor operations occur.
    result = tf.compat.v1.linalg.tensor_diag_part(input_tensor_bf16)

    # Define the expected output in bfloat16
    expected_output = tf.constant([1.5, 2.5, 3.5], dtype=tf.bfloat16)

    # Assert that the result matches the expected output.
    # If the API suffered from the same "garbage" issue with this dtype,
    # this assertion would fail.
    assert tf.reduce_all(tf.equal(result, expected_output)).numpy(), \
        "tf.compat.v1.linalg.tensor_diag_part returned garbage/incorrect results for bfloat16"

    print("Test passed: Operation successful with bfloat16 dtype.")

if __name__ == "__main__":
    test_tensor_diag_part_bfloat16_fix()