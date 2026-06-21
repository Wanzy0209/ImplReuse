import torch
import tensorflow as tf
import numpy as np

def test_gather_does_not_modify_global_backend_state():
    """
    Test that tf.keras.backend.gather does not inadvertently modify global
    backend settings. This mirrors the PyTorch issue where torch.compile
    (used in a similar context) modified global distribution validation args.
    """
    # Capture initial global state of the Keras backend
    original_floatx = tf.keras.backend.floatx()
    original_epsilon = tf.keras.backend.epsilon()
    original_image_data_format = tf.keras.backend.image_data_format()

    # Setup data for the gather operation
    data = tf.constant([[1, 2], [3, 4], [5, 6]])
    indices = tf.constant([0, 2])

    # Call the API under test
    # In the PyTorch issue, the compilation of a mask triggered the side effect.
    # Here we test the gather operation which calls gen_xla_ops.xla_gather.
    result = tf.keras.backend.gather(data, indices)

    # Verify that global state has not changed
    assert tf.keras.backend.floatx() == original_floatx, \
        "tf.keras.backend.gather modified global floatx setting."
    assert tf.keras.backend.epsilon() == original_epsilon, \
        "tf.keras.backend.gather modified global epsilon setting."
    assert tf.keras.backend.image_data_format() == original_image_data_format, \
        "tf.keras.backend.gather modified global image_data_format setting."

    # Verify the operation result is correct
    expected = np.array([[1, 2], [5, 6]])
    np.testing.assert_array_equal(result.numpy(), expected)

if __name__ == "__main__":
    test_gather_does_not_modify_global_backend_state()
    print("Test passed: No global side effects detected.")