import tensorflow as tf
import numpy as np

def test_categorical_crossentropy():
    """
    Test case for tf.keras.losses.CategoricalCrossentropy.
    
    Context: The original issue (PyTorch #160980) reported a build failure 
    (undefined references) when using NVSHMEM with CUDA 13 on SM_75 architecture.
    This test verifies the functional correctness of the semantically similar 
    API (CategoricalCrossentropy) to ensure the loss calculation logic is 
    properly linked and executes without runtime errors, analogous to ensuring 
    the distributed component links correctly.
    """
    
    # Setup data matching the API documentation examples
    y_true = [[0, 1, 0], [0, 0, 1]]
    y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    
    # Initialize the loss function
    cce = tf.keras.losses.CategoricalCrossentropy()
    
    # Test 1: Basic calculation
    # Expected value from docs: 1.177
    loss = cce(y_true, y_pred)
    assert np.isclose(loss.numpy(), 1.177, atol=1e-3), \
        f"Test 1 Failed: Expected ~1.177, got {loss.numpy()}"
    
    # Test 2: Calculation with sample_weight
    # Expected value from docs: 0.814
    sample_weight = tf.constant([0.3, 0.7])
    loss_weighted = cce(y_true, y_pred, sample_weight=sample_weight)
    assert np.isclose(loss_weighted.numpy(), 0.814, atol=1e-3), \
        f"Test 2 Failed: Expected ~0.814, got {loss_weighted.numpy()}"
        
    # Test 3: Different reduction type (SUM)
    cce_sum = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM)
    loss_sum = cce_sum(y_true, y_pred)
    # Verify it runs and returns a scalar
    assert loss_sum.shape == (), "Test 3 Failed: Sum reduction did not return scalar"
    
    print("All tests passed for CategoricalCrossentropy.")

if __name__ == "__main__":
    test_categorical_crossentropy()