import torch
import tensorflow as tf

def test_tf_keras_backend_one_hot_4d():
    """
    Adapted test case based on PyTorch Issue 160053.
    Original Bug: torch.nn.functional.pad(mode="circular") fails for 4D input 
    despite the error message claiming support for 4D and 5D.
    
    This test verifies the behavior of the similar API (tf.keras.backend.one_hot) 
    when provided with a 4D input tensor, checking if it handles the dimensionality 
    as documented (nD input -> (n+1)D output).
    """
    
    # Create a 4D input tensor, matching the dimensionality of the PyTorch bug report
    # PyTorch: a = torch.empty(2,2,2,2)
    # TensorFlow one_hot requires integer indices
    indices = tf.ones((2, 2, 2, 2), dtype=tf.int32)
    
    num_classes = 3
    
    # Call the similar API
    # PyTorch: F.pad(a, (1,1), mode="circular")
    # TensorFlow: one_hot(indices, num_classes)
    try:
        result = tf.keras.backend.one_hot(indices, num_classes)
        
        # Verify the output shape. 
        # Documentation states: Returns (n + 1)D one hot representation.
        # Input is 4D, so output should be 5D.
        expected_shape = (2, 2, 2, 2, num_classes)
        
        assert result.shape == expected_shape, (
            f"Expected shape {expected_shape} for 4D input, but got {result.shape}. "
            "This indicates a potential issue with dimension handling similar to the PyTorch bug."
        )
        
        print("Test Passed: tf.keras.backend.one_hot correctly handles 4D input.")
        
    except Exception as e:
        print(f"Test Failed with error: {e}")
        raise

if __name__ == "__main__":
    test_tf_keras_backend_one_hot_4d()