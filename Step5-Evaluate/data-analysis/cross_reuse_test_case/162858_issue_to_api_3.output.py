import tensorflow as tf

# Enable eager execution to match the 'backend="eager"' context from the PyTorch issue
tf.compat.v1.enable_eager_execution()

def test_global_step_eager_context():
    """
    Test case for tf.compat.v1.train.global_step in eager mode.
    This mirrors the PyTorch issue's focus on behavior within an eager execution context.
    """
    # Create a variable to hold the global step
    initial_value = 10
    global_step_tensor = tf.Variable(initial_value, trainable=False, name='global_step')

    # Call the API. In eager mode, the session argument is effectively ignored
    # (or should be None), and it relies on the tensor's value directly.
    # This corresponds to the 'backend="eager"' usage in the original issue.
    step_value = tf.compat.v1.train.global_step(None, global_step_tensor)

    # Verify the result matches the expected behavior in eager mode
    assert step_value == initial_value, \
        f"Expected global step to be {initial_value}, but got {step_value}"

    print("Test passed: Global step retrieved correctly in eager mode.")

if __name__ == "__main__":
    test_global_step_eager_context()