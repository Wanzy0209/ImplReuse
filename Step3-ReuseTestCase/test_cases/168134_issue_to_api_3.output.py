import tensorflow as tf
import numpy as np

def test_masking_uneven_gaps():
    """
    Test case for tf.keras.layers.Masking inspired by Issue 168134.
    
    The original bug involved incorrect handling of 'pad' elements in an uneven 
    strided shard distribution. This test verifies that the Masking layer 
    correctly identifies and handles 'gaps' (masked timesteps) in a sequence,
    ensuring the mask logic correctly processes uneven data patterns similar 
    to the shard distribution issue.
    """
    # Setup data similar to the bug report's [0, 1, 2, 3, 4]
    # Shape: (batch_size=1, timesteps=5, features=1)
    data = np.array([[[0.], [1.], [2.], [3.], [4.]]], dtype=np.float32)

    # Simulate the "gaps" or "pads" mentioned in the bug report.
    # In the bug, the sharding logic failed to place the 'pad' correctly.
    # Here, we explicitly introduce gaps at indices 2 and 4 to test 
    # the Masking layer's ability to identify these uneven spots.
    # We use 0.0 as the mask_value (standard for Masking layer).
    data[0, 2, :] = 0.0
    data[0, 4, :] = 0.0

    # Initialize the Masking layer
    # mask_value=0.0 will treat the zeros we just set as gaps
    masking_layer = tf.keras.layers.Masking(mask_value=0.0)

    # Compute the mask
    mask = masking_layer.compute_mask(data)

    # Define the expected mask.
    # Indices 0, 1, 3 contain valid data (True).
    # Indices 2, 4 contain the mask value/gaps (False).
    expected_mask = np.array([[True, True, False, True, False]])

    # Assert that the mask correctly identifies the gaps
    # This mirrors the bug report's check for correct local tensor content
    assert np.array_equal(mask.numpy(), expected_mask), \
        f"Mask mismatch. Expected {expected_mask}, got {mask.numpy()}"

if __name__ == "__main__":
    test_masking_uneven_gaps()
    print("Test passed: Masking layer correctly handled uneven gaps.")