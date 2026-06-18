import tensorflow as tf

# Disable eager execution to test the graph mode path of local_variables_initializer,
# which corresponds to the 'variables_initializer(local_variables())' logic in the API.
tf.compat.v1.disable_eager_execution()

def test_complex_local_variable_initialization():
    """
    Test case derived from PyTorch Issue 160034.
    
    The original issue highlights a failure mode where index_put_ does not support 
    complex value insertion on the MPS backend, resulting in a cryptic error.
    
    This test verifies that tf.compat.v1.local_variables_initializer handles 
    complex64 variables correctly, ensuring that initialization succeeds 
    (or fails with a clear message) for this data type.
    """
    with tf.compat.v1.Session() as sess:
        # Create a local variable with complex64, mirroring the bug report's setup
        # where 'image' and 'data' were complex64.
        complex_var = tf.compat.v1.get_variable(
            "complex_local_var",
            shape=[3],
            dtype=tf.complex64,
            initializer=tf.ones_initializer(dtype=tf.complex64)
        )

        # Retrieve the initializer operation for local variables
        init_op = tf.compat.v1.local_variables_initializer()

        # Run the initialization
        sess.run(init_op)

        # Verify the variable was initialized correctly with the expected dtype and value
        result = sess.run(complex_var)
        
        # Assertions to check behavior
        assert result.dtype == tf.complex64, "Variable dtype should be complex64"
        # Check if values are initialized to 1+0j (matching ones_initializer)
        assert tf.reduce_all(result == 1+0j).numpy(), "Variable values should be initialized to 1+0j"

if __name__ == "__main__":
    test_complex_local_variable_initialization()
    print("Test passed: Complex local variables initialized successfully.")