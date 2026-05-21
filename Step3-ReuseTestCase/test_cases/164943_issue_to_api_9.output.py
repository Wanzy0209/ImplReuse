import tensorflow as tf
import numpy as np

def test_zero_padding_3d_basic():
    """
    Test case for tf.keras.layers.ZeroPadding3D based on the usage pattern 
    found in the similar API information.
    
    This test verifies that the layer correctly applies padding to 3D input data,
    mirroring the structure of the original bug report (instantiating data and 
    applying a transformation).
    """
    # Setup input data matching the similar API example
    input_shape = (1, 1, 2, 2, 3)
    x = np.arange(np.prod(input_shape)).reshape(input_shape).astype('float32')
    
    # Apply the ZeroPadding3D layer
    # The original issue involved a backend check ('mps'). 
    # Here we test the layer's functionality which is the core of the similar API.
    padding = 2
    layer = tf.keras.layers.ZeroPadding3D(padding=padding)
    y = layer(x)
    
    # Calculate expected shape
    # Input: (1, 1, 2, 2, 3)
    # Padding: 2 on each side of the 3 spatial dimensions
    # Expected: (1, 1+4, 2+4, 2+4, 3) -> (1, 5, 6, 6, 3)
    expected_shape = (1, 5, 6, 6, 3)
    
    # Assertions
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, but got {y.shape}"
    
    # Verify that the padding is actually zeros (checking corners)
    # The original data should be in the center
    assert np.array_equal(y[0, 2:-2, 2:-2, 2:-2, :], x), "Center data does not match input"
    assert np.all(y[0, 0, 0, 0, :] == 0), "Padding corners are not zero"
    
    print("Test passed: ZeroPadding3D applied correctly.")

if __name__ == "__main__":
    test_zero_padding_3d_basic()