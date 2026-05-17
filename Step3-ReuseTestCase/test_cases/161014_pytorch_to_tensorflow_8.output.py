import torch
import tensorflow as tf
import numpy as np

def test_one_hot_negative_size():
    """
    Adapted test case for tf.keras.preprocessing.text.one_hot based on 
    PyTorch constant_pad_nd negative padding bug (Issue 161014).
    
    Original Logic: Passing negative padding values to constant_pad_nd 
    results in inconsistent behavior (RuntimeError vs valid output).
    
    Adaptation: In one_hot, the parameter controlling the output dimension 
    size is 'num_classes'. We test the behavior when 'num_classes' is 
    negative or zero, analogous to negative padding reducing tensor size.
    """
    
    # Input tensor (indices must be integers for one_hot)
    # Analogous to torch.ones([5, 3])
    indices = np.array([[0, 1, 2], [1, 2, 0]])

    # Test Case 1: Negative num_classes (Analogous to negative padding)
    # PyTorch behavior: Inconsistent (sometimes works, sometimes RuntimeError).
    # TensorFlow behavior: Expected to raise error for invalid depth.
    print("Testing with num_classes = -1 (Negative)...")
    try:
        # Note: Using the signature provided in the Similar API information
        result = tf.keras.preprocessing.text.one_hot(indices, num_classes=-1)
        print(f"Result Shape: {result.shape}")
        print("Test Passed: API accepted negative num_classes.")
    except Exception as e:
        print(f"Test Failed/Exception: {type(e).__name__}: {e}")

    # Test Case 2: Zero num_classes (Analogous to padding resulting in size 0)
    # PyTorch behavior: Often results in size 0 dimension.
    print("\nTesting with num_classes = 0 (Zero)...")
    try:
        result = tf.keras.preprocessing.text.one_hot(indices, num_classes=0)
        print(f"Result Shape: {result.shape}")
        print("Test Passed: API accepted zero num_classes.")
    except Exception as e:
        print(f"Test Failed/Exception: {type(e).__name__}: {e}")

    # Test Case 3: Valid positive num_classes (Control)
    print("\nTesting with num_classes = 3 (Valid)...")
    try:
        result = tf.keras.preprocessing.text.one_hot(indices, num_classes=3)
        print(f"Result Shape: {result.shape}")
        print("Test Passed: API behaved as expected.")
    except Exception as e:
        print(f"Test Failed/Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_one_hot_negative_size()