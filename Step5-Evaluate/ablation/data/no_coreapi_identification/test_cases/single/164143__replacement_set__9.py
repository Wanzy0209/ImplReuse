import torch
import tensorflow as tf
import numpy as np

def test_he_normal_with_tanh():
    """
    Test case generated based on Issue 164143 logic.
    
    Original Logic: Ensure that a specific mode/mechanism (DebugMode/torch.compile) 
    does not silently disable or skip the intended operation.
    
    Adaptation: 
    - Source APIs (torch.compile, DebugMode) are replaced with tf.keras.initializers.HeNormal.
    - Similar API (torch.nn.functional.tanh) is used as the activation function.
    - The test verifies that HeNormal initializes weights correctly (non-zero, expected distribution)
      rather than "silently failing" (e.g., defaulting to zeros).
    """
    
    # Replacement API: tf.keras.initializers.HeNormal
    # This replaces the combination of torch.compile and DebugMode as the primary subject of the test.
    initializer = tf.keras.initializers.HeNormal()
    
    # Similar API: torch.nn.functional.tanh
    # Leveraged here as the activation function for the layer.
    layer = tf.keras.layers.Dense(
        units=10, 
        activation='tanh', 
        kernel_initializer=initializer
    )
    
    # Build the layer to trigger initialization
    layer.build((None, 5))
    
    # Get the initialized weights
    weights = layer.get_weights()[0]
    
    # Assertion 1: Check that the weights are not all zeros.
    # This corresponds to checking that the operation didn't "silently disable" or fail to initialize.
    assert not np.allclose(weights, 0), "HeNormal silently failed to initialize weights (all zeros)"
    
    # Assertion 2: Check statistical properties of HeNormal.
    # HeNormal draws samples from a truncated normal distribution centered on 0 
    # with stddev = sqrt(2 / fan_in). Here fan_in = 5.
    # stddev = sqrt(2 / 5) = sqrt(0.4) ~= 0.632
    expected_stddev = np.sqrt(2 / 5)
    actual_stddev = np.std(weights)
    
    # Allow for some tolerance due to random sampling
    assert np.isclose(actual_stddev, expected_stddev, atol=0.2), \
        f"Weight distribution deviation too high: expected ~{expected_stddev}, got {actual_stddev}"

    print("Test passed: HeNormal initialized weights correctly with tanh activation.")

if __name__ == "__main__":
    test_he_normal_with_tanh()