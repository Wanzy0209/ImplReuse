import torch
import tensorflow as tf

def test_name_scope_no_global_side_effects():
    """
    Test to verify that tf.keras.backend.name_scope does not have unintended
    global side effects, similar to the PyTorch bug where torch.compile
    inadvertently changed global distribution validation settings.
    """
    # Capture initial global state of the Keras backend
    original_floatx = tf.keras.backend.floatx()
    original_epsilon = tf.keras.backend.epsilon()
    original_image_data_format = tf.keras.backend.image_data_format()

    # Execute the API under test
    with tf.keras.backend.name_scope("test_scope"):
        # Perform an operation to ensure the scope is active and used
        v = tf.Variable(1.0, name="my_var")
        
        # Verify intended behavior: name is scoped correctly
        assert v.name == "test_scope/my_var:0", f"Name scoping failed: {v.name}"

        # Verify no unintended global side effects occurred inside the scope
        # (mimicking the check for torch.distributions.Distribution.set_default_validate_args)
        current_floatx = tf.keras.backend.floatx()
        current_epsilon = tf.keras.backend.epsilon()
        current_image_data_format = tf.keras.backend.image_data_format()

        assert current_floatx == original_floatx, \
            f"Global floatx changed from {original_floatx} to {current_floatx}"
        assert current_epsilon == original_epsilon, \
            f"Global epsilon changed from {original_epsilon} to {current_epsilon}"
        assert current_image_data_format == original_image_data_format, \
            f"Global image_data_format changed from {original_image_data_format} to {current_image_data_format}"

    # Verify global state remains unchanged after exiting the scope
    assert tf.keras.backend.floatx() == original_floatx
    assert tf.keras.backend.epsilon() == original_epsilon
    assert tf.keras.backend.image_data_format() == original_image_data_format

if __name__ == "__main__":
    test_name_scope_no_global_side_effects()
    print("Test passed: No unintended global side effects detected.")