import tensorflow as tf
import numpy as np

def test_categorical_crossentropy_basic():
    """
    Test case for tf.keras.metrics.CategoricalCrossentropy based on the 
    usage pattern found in the similar API information.
    
    This test verifies the correct computation of crossentropy loss 
    with one-hot encoded labels, ensuring the API behaves as documented.
    """
    # Setup data from the similar API information
    y_true = [[0, 1, 0], [0, 0, 1]]
    y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

    # Test 1: Using 'auto'/'sum_over_batch_size' reduction type
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(y_true, y_pred).numpy()
    
    # Assert the result matches the expected value from the documentation
    expected_loss = 1.177
    assert np.isclose(loss, expected_loss, atol=1e-3), \
        f"Expected loss {expected_loss}, but got {loss}"

    # Test 2: Calling with 'sample_weight'
    sample_weight = tf.constant([0.3, 0.7])
    weighted_loss = cce(y_true, y_pred, sample_weight=sample_weight).numpy()
    
    # Assert the weighted result matches the expected value
    expected_weighted_loss = 0.814
    assert np.isclose(weighted_loss, expected_weighted_loss, atol=1e-3), \
        f"Expected weighted loss {expected_weighted_loss}, but got {weighted_loss}"

    # Test 3: Using 'sum' reduction type
    # Note: The original snippet cut off, but we verify the configuration works
    cce_sum = tf.keras.losses.CategoricalCrossentropy(reduction='sum')
    sum_loss = cce_sum(y_true, y_pred).numpy()
    
    # Manually calculate expected sum for verification
    # Loss per sample: -sum(y_true * log(y_pred))
    # Sample 1: -(0*log(0.05) + 1*log(0.95) + 0*log(0)) = -log(0.95)  0.0513
    # Sample 2: -(0*log(0.1) + 0*log(0.8) + 1*log(0.1)) = -log(0.1)  2.3026
    # Sum  2.3539
    expected_sum = 2.3539
    assert np.isclose(sum_loss, expected_sum, atol=1e-3), \
        f"Expected sum loss {expected_sum}, but got {sum_loss}"

if __name__ == "__main__":
    test_categorical_crossentropy_basic()
    print("Test passed.")