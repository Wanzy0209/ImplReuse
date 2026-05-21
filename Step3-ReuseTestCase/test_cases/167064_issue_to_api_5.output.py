import torch
import tensorflow as tf
import numpy as np

def test_tf_keras_backend_relu_no_global_side_effects():
    """
    Test that tf.keras.backend.relu does not modify global backend state.
    This mirrors the PyTorch issue where torch.compile inadvertently
    changed global distribution settings (set_default_validate_args).
    """
    # Capture initial global state of the Keras backend
    initial_floatx = tf.keras.backend.floatx()
    initial_epsilon = tf.keras.backend.epsilon()
    initial_image_data_format = tf.keras.backend.image_data_format()

    # Create input tensor
    x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Call the API (tf.keras.backend.relu)
    # The similar API code pattern shows it uses context.get_default()
    y = tf.keras.backend.relu(x)

    # Verify the functional correctness
    expected = tf.constant([0.0, 0.0, 0.0, 1.0, 2.0])
    assert np.array_equal(y.numpy(), expected.numpy()), "ReLU calculation incorrect"

    # Verify that global state has not been modified
    # This addresses the core concern of the original bug: unintended global side effects
    assert tf.keras.backend.floatx() == initial_floatx, \
        "tf.keras.backend.relu modified global floatx setting"
    assert tf.keras.backend.epsilon() == initial_epsilon, \
        "tf.keras.backend.relu modified global epsilon setting"
    assert tf.keras.backend.image_data_format() == initial_image_data_format, \
        "tf.keras.backend.relu modified global image_data_format setting"

if __name__ == "__main__":
    test_tf_keras_backend_relu_no_global_side_effects()
    print("Test passed.")