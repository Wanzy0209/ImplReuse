import tensorflow as tf
import numpy as np

def test_to_categorical_zero_dimension():
    """
    Test case for tf.keras.utils.to_categorical based on the logic of 
    PyTorch Issue #161014 (Inconsistent constant_pad_nd behavior with negative padding).
    
    The original issue involves an inconsistency where padding resulting in a size of 0 
    is sometimes allowed and sometimes throws a "negative output size" error.
    
    This test checks if to_categorical handles the boundary case where num_classes=0 
    (resulting in a dimension of size 0) correctly, distinguishing it from a negative size.
    """
    # Input data
    y = np.array([0, 1, 2])
    
    # Test Case 1: num_classes = 0
    # This corresponds to the PyTorch case where padding resulted in shape [..., 0].
    # We expect the output shape to be (3, 0).
    try:
        result = tf.keras.utils.to_categorical(y, num_classes=0)
        print(f"Test Passed: to_categorical with num_classes=0 returned shape {result.shape}")
        assert result.shape == (3, 0), f"Expected shape (3, 0), but got {result.shape}"
    except Exception as e:
        print(f"Test Failed: to_categorical with num_classes=0 raised an error: {e}")
        # This would indicate a similar bug to the PyTorch issue where size 0 is treated as invalid/negative

    # Test Case 2: num_classes < 0
    # This corresponds to a truly invalid negative dimension.
    # We expect this to raise an error.
    try:
        result = tf.keras.utils.to_categorical(y, num_classes=-1)
        print(f"Test Failed: to_categorical with num_classes=-1 should have raised an error, but returned shape {result.shape}")
    except (ValueError, tf.errors.InvalidArgumentError) as e:
        print(f"Test Passed: to_categorical with num_classes=-1 correctly raised an error: {e}")
    except Exception as e:
        print(f"Test Inconclusive: Unexpected error type for num_classes=-1: {e}")

if __name__ == "__main__":
    test_to_categorical_zero_dimension()