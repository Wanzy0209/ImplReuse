import numpy as np
import tensorflow as tf

def test_cropping3d_layer():
    """
    Test case for tf.keras.layers.Cropping3D based on the provided usage pattern.
    This mirrors the structure of the original bug report (create data -> apply operation)
    but applies it to the similar API.
    """
    # Setup input data as per the similar API example
    input_shape = (2, 28, 28, 10, 3)
    x = np.arange(np.prod(input_shape)).reshape(input_shape)

    # Apply the Cropping3D layer
    # This corresponds to the 'foo.to('mps')' operation in the original issue
    y = tf.keras.layers.Cropping3D(cropping=(2, 4, 2))(x)

    # Verify the output shape matches the expected result
    # Original issue expected a tensor on MPS, here we expect a specific shape
    expected_shape = (2, 24, 20, 6, 3)
    assert y.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {y.shape}"

if __name__ == "__main__":
    test_cropping3d_layer()