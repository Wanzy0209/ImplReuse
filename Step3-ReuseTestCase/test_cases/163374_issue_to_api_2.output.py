import tensorflow as tf
import numpy as np

def test_masking_layer_inplace_modification():
    """
    Test case for tf.keras.layers.Masking inspired by the PyTorch DTensor 
    inplace ops bug (Issue 163374).
    
    The original bug involves an inplace operation (clamp_) modifying a tensor 
    and resulting in incorrect state (placement/value). This test verifies that 
    tf.keras.layers.Masking correctly handles inputs that have been modified 
    inplace (set to mask_value) and produces the correct mask state.
    """
    # 1. Setup: Create a sample input tensor
    # Mimicking the tensor creation in the bug report
    samples, timesteps, features = 2, 5, 3
    inputs = np.random.random([samples, timesteps, features]).astype(np.float32)

    # 2. Inplace modification: Set specific timesteps to the mask value (0.0)
    # This mirrors the 'inplace ops' aspect of the bug report (clamp_)
    # In the bug, values are clamped. Here, we set them to 0 to trigger masking.
    inputs[:, 2, :] = 0.0
    inputs[:, 4, :] = 0.0

    # 3. Instantiate the Masking layer
    # This is the "Similar API" we are testing
    masking_layer = tf.keras.layers.Masking(mask_value=0.0)

    # 4. Apply the layer
    output = masking_layer(inputs)

    # 5. Assertions
    # Check that the output tensor shape is preserved
    assert output.shape == (samples, timesteps, features), \
        f"Shape mismatch: expected {(samples, timesteps, features)}, got {output.shape}"

    # Check that the mask is correctly generated based on the inplace modification
    # The mask should be True (masked) for timesteps 2 and 4
    expected_mask = np.array([[False, False, True, False, True],
                               [False, False, True, False, True]])
    
    # Access the computed mask from the output tensor
    computed_mask = output._keras_mask.numpy()
    
    np.testing.assert_array_equal(computed_mask, expected_mask, 
        err_msg="Mask was not generated correctly after inplace modification.")

    # Check that the data values are not modified by the layer itself
    # (Masking layer passes data through, only the mask is added)
    np.testing.assert_array_equal(output.numpy(), inputs,
        err_msg="Output tensor values were unexpectedly modified.")

if __name__ == '__main__':
    test_masking_layer_inplace_modification()
    print("Test passed: tf.keras.layers.Masking correctly handles inplace modifications.")