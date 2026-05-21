import torch
import tensorflow as tf

def test_yiq_to_rgb_zero_shape():
    """
    Adapted test case based on PyTorch issue #168071.
    Original issue: torch.nn.functional.pad crashes when padding (0, 0) to 0-shape.
    Original input shape: (6, 0).
    
    This test checks if tf.image.yiq_to_rgb handles tensors with a 0-dimension 
    gracefully. The API requires the last dimension to be 3, so we adapt the 
    shape to (6, 0, 3) to preserve the core logic of testing a 0-dimension 
    tensor while satisfying API constraints.
    """
    # Create a tensor with a 0-dimension (similar to the original (6, 0))
    # Last dim is 3 to satisfy tf.image.yiq_to_rgb requirements
    x0 = tf.zeros((6, 0, 3))

    try:
        # Perform the operation
        y1 = tf.image.yiq_to_rgb(x0)
        
        # Verify the output shape is preserved and valid
        # Expected: (6, 0, 3) - operation should not crash on 0-shape
        print(f"Test Passed. Output shape: {y1.shape}")
        assert y1.shape == (6, 0, 3), f"Expected shape (6, 0, 3), but got {y1.shape}"
        
    except Exception as e:
        print(f"Test Failed with error: {e}")
        raise

if __name__ == "__main__":
    test_yiq_to_rgb_zero_shape()