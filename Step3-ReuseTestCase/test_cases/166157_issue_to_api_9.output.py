import tensorflow as tf
import numpy as np

def test_exp_basic_functionality_and_observability():
    """
    Test case for tf.keras.ops.exp inspired by Issue 166157.
    
    The issue highlights the necessity of moving from a "black box" implementation
    (basic functionality only) to one that provides essential observability.
    
    This test verifies that tf.keras.ops.exp:
    1. Performs the basic mathematical operation correctly (Basic Functionality).
    2. Returns results that are inspectable and predictable (Observability).
    """
    # Define input data
    input_data = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    
    # Execute the operation
    # This mirrors the 'allocate' call in the issue, but for a math operation.
    result = tf.keras.ops.exp(input_data)
    
    # 1. Verify Basic Functionality
    # Ensure the operation performs the core task (calculating e^x) correctly.
    expected_output = np.exp([1.0, 2.0, 3.0])
    np.testing.assert_allclose(result.numpy(), expected_output, rtol=1e-5)
    
    # 2. Verify Observability
    # The issue complains about "zero visibility into memory consumption".
    # For a math operation, visibility implies we can verify the properties of the output.
    assert result.shape == input_data.shape, "Shape should be preserved for observability"
    assert result.dtype == input_data.dtype, "Dtype should be preserved for observability"
    
    # Ensure the result is not a "black box" by checking specific value properties
    # (e.g., exp(x) is always positive for real numbers).
    assert tf.reduce_all(result > 0).numpy(), "Exponential of real numbers should be positive"

if __name__ == "__main__":
    test_exp_basic_functionality_and_observability()
    print("Test passed.")