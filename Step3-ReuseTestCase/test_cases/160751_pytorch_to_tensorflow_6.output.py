import tensorflow as tf

def func():
    # Create a tensor
    a = tf.constant([1.0, -2.0])
    # Perform operation
    result = tf.reduce_all(a > 0)
    
    # Use the similar API: tf.compat.v1.name_scope
    # We set skip_on_eager=False to ensure the scope is active in eager mode,
    # analogous to the specific backend mode in the original bug.
    with tf.compat.v1.name_scope("test_scope", skip_on_eager=False):
        # The original code asserts result (which is False), expecting a throw.
        # We use .numpy() to get the value for the Python assert.
        assert result.numpy() == False, "should throw"
        
        # This should not run if the assertion works correctly
        print("should not run")

def test_fn():
    try:
        func()
        print("Test Failed: Exception was not caught")
    except AssertionError as e:
        print(f"Test Passed: Exception caught as expected - {e}")

if __name__ == "__main__":
    test_fn()